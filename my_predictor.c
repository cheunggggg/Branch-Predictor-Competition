#include <string.h>
#define TAGE_TABLES 5
#define TABLE_SIZE 131072
#define PERC_SIZE 4194304
#define PERC_HIST 64
#define PERC_THETA 109
#define LOOP_SIZE 16384
#define MAX_ITER 1023
#define SC_TABLES 4
#define SC_SIZE 131072
#define LOCAL_HIST_SIZE 32768
#define LOCAL_HIST_LEN 17
#define BIAS_SIZE 131072
#define META_SIZE 4194304
#define STABILITY_SIZE 16384

const int hist_lens[TAGE_TABLES] = {0, 1, 8, 30, 67}; // TAGE history lengths
const int tag_bits_arr[TAGE_TABLES] = {7, 9, 11, 12, 13}; // Tag bits per bank

struct tage_entry {
    unsigned short tag;
    unsigned char ctr;
    unsigned char u;
    tage_entry() : tag(0), ctr(0), u(0) {}
};
struct loop_entry {
    unsigned short tag;
    unsigned short iter;
    unsigned short cur_iter;
    unsigned char conf;
    loop_entry() : tag(0), iter(0), cur_iter(0), conf(0) {}
};
struct local_hist_entry {
    unsigned int hist;
    local_hist_entry() : hist(0) {}
};
class my_update : public branch_update {
public:
    unsigned int pc;
    int hit_bank, alt_bank;
    unsigned int indices[TAGE_TABLES];
    unsigned short tags[TAGE_TABLES];
    bool tage_pred, alt_pred, provider_pred;
    unsigned int perc_idx;
    int perc_sum;
    bool perc_pred;
    unsigned int sc_indices[SC_TABLES];
    int sc_sums[SC_TABLES];
    int sc_total;
    unsigned int local_idx;
    unsigned int local_hist;
    int local_sum;
    unsigned int bias_idx;
    int bias_sum;
    unsigned int meta_idx;
    unsigned char meta_ctr;
    unsigned int loop_idx;
    unsigned int base_idx;
    bool provider_weak;
    unsigned int stab_idx;
    bool tage_stable;
};
class my_predictor : public branch_predictor {
public:
    tage_entry tage[TAGE_TABLES][TABLE_SIZE]; // TAGE predictor (2.5MB)
    loop_entry loops[LOOP_SIZE]; // Loop predictor (0.11MB)
    signed char perc_weights[PERC_SIZE][PERC_HIST + 1]; // Perceptron weights (260MB)
    signed char sc[SC_TABLES][SC_SIZE]; // Statistical corrector (0.5MB)
    local_hist_entry local_hist[LOCAL_HIST_SIZE]; // Local history register (0.13MB)
    signed char local_pred[LOCAL_HIST_SIZE][(1 << LOCAL_HIST_LEN)]; // Local predictor table (4GB)
    signed char bias[BIAS_SIZE]; // Bias weights (0.13MB)
    unsigned char meta[META_SIZE]; // Meta-predictor selector (4MB)
    unsigned char stability[STABILITY_SIZE]; // TAGE stability tracker (16KB)
    unsigned long long ghist; // Global history register
    unsigned long long path_hist; // Path history register
    unsigned char base[TABLE_SIZE]; // Base 2-bit predictor (128KB)
    unsigned long long branch_count;
    my_update u;
    branch_info bi;
    my_predictor() : ghist(0), path_hist(0), branch_count(0) {
        memset(base, 2, sizeof(base));
        for (int b = 0; b < TAGE_TABLES; b++) {
            unsigned init_val = (b >= 2) ? 4 : 2;
            for (int i = 0; i < TABLE_SIZE; i++) {
                tage[b][i].tag = 0;
                tage[b][i].ctr = init_val;
                tage[b][i].u = 0;
            }
        }
        memset(loops, 0, sizeof(loops));
        memset(perc_weights, 0, sizeof(perc_weights));
        memset(sc, 0, sizeof(sc));
        memset(local_hist, 0, sizeof(local_hist));
        memset(local_pred, 0, sizeof(local_pred));
        memset(bias, 0, sizeof(bias));
        memset(meta, 2, sizeof(meta));
        memset(stability, 2, sizeof(stability));
    }
    // Fold history bits into smaller hash value
    static inline unsigned fold(unsigned long long val, unsigned len, unsigned bits) {
        if (!bits) return 0;
        const unsigned long long input_mask = (len >= 64) ? ~0ull : ((1ull << len) - 1ull);
        const unsigned long long masked_val = val & input_mask;
        const unsigned chunk_mask = (1u << bits) - 1u;
        unsigned result = 0;
        unsigned offset = 0;
        unsigned double_bits = bits << 1;
        while (offset + double_bits <= len) {
            result = (result + static_cast<unsigned>((masked_val >> offset) & chunk_mask)) & chunk_mask;
            result = (result + static_cast<unsigned>((masked_val >> (offset + bits)) & chunk_mask)) & chunk_mask;
            offset += double_bits;
        }
        while (offset < len) {
            result = (result + static_cast<unsigned>((masked_val >> offset) & chunk_mask)) & chunk_mask;
            offset += bits;
        }
        return result & chunk_mask;
    }
    // Compute TAGE table index using PC, global history, and path history
    unsigned int hash_idx(unsigned int pc, unsigned long long h, int len, int bank) {
        unsigned fh1 = fold(ghist, len, 17);
        unsigned fh2 = fold((ghist >> (bank + 1)) ^ (ghist << (bank + 3)), len > 16 ? 16 : len, 17);
        unsigned fpath = fold(path_hist ^ (path_hist >> (bank + 2)), len, 17);
        unsigned mixpc = (pc ^ ((pc >> 11) | (pc << 5))) ^ (bank * 0xDEADBEEF);
        unsigned mix1 = (fh1 * 0x9E3779B9) ^ (fh2 * 0x85EBCA6B);
        unsigned mix2 = (fpath * 0xC2B2AE3D) ^ (mixpc * 0x27D4EB2D);
        return (mix1 ^ mix2 ^ (fh1 << bank) ^ (fh2 >> bank)) & (TABLE_SIZE - 1);
    }
    // Compute TAGE entry tag
    unsigned short hash_tag(unsigned int pc, unsigned long long h, int len, int bank) {
        unsigned tbits = tag_bits_arr[bank];
        unsigned f1 = fold(ghist, len, tbits);
        unsigned f2 = fold(ghist * 0x123456789ABCDEFULL, len, tbits);
        unsigned fpath = fold(path_hist, len > 32 ? 32 : len, tbits);
        return (pc ^ (f1 << 1) ^ f2 ^ fpath) & ((1u << tbits) - 1u);
    }
    branch_update *predict(branch_info &b) {
        bi = b;
        u.pc = b.address >> 2;
        u.hit_bank = -1;
        u.alt_bank = -1;
        u.provider_weak = false;
        u.tage_stable = true;
        if (!(b.br_flags & BR_CONDITIONAL)) {
            u.direction_prediction(true);
            return &u;
        }
        u.base_idx = u.pc & (TABLE_SIZE - 1);
        bool base_pred = (base[u.base_idx] >> 1) != 0;
        // Search TAGE tables for matching tags (longest history first)
        for (int i = TAGE_TABLES - 1; i >= 0; i--) {
            u.indices[i] = hash_idx(u.pc, ghist, hist_lens[i], i);
            u.tags[i] = hash_tag(u.pc, ghist, hist_lens[i], i);
            if (tage[i][u.indices[i]].tag == u.tags[i]) {
                if (u.hit_bank == -1) {
                    u.hit_bank = i;
                } else if (u.alt_bank == -1) {
                    u.alt_bank = i;
                    break;
                }
            }
        }
        // Get TAGE prediction (use altpred if provider weak and not useful)
        bool final_tage = base_pred;
        if (u.hit_bank >= 0) {
            unsigned char pctr = tage[u.hit_bank][u.indices[u.hit_bank]].ctr;
            unsigned ctr_mid = (u.hit_bank >= 2) ? 4 : 2;
            u.provider_pred = pctr >= ctr_mid;
            unsigned char actr = (u.alt_bank >= 0) ? tage[u.alt_bank][u.indices[u.alt_bank]].ctr : 0;
            unsigned alt_mid = (u.alt_bank >= 2) ? 4 : 2;
            u.alt_pred = (u.alt_bank >= 0) ? (actr >= alt_mid) : base_pred;
            bool pWeak = (u.hit_bank >= 2) ? (pctr >= 3 && pctr <= 5) : (pctr == 1 || pctr == 2);
            bool notUseful = (tage[u.hit_bank][u.indices[u.hit_bank]].u == 0);
            u.provider_weak = pWeak;
            final_tage = ((pWeak && notUseful) ? u.alt_pred : u.provider_pred);
        } else {
            u.provider_pred = base_pred;
            u.alt_pred = base_pred;
        }
        u.stab_idx = (u.pc ^ (u.pc >> 5)) & (STABILITY_SIZE - 1);
        u.tage_stable = (stability[u.stab_idx] >= 2);
        // Apply loop predictor override
        u.loop_idx = (u.pc ^ (u.pc >> 6)) & (LOOP_SIZE - 1);
        loop_entry *lp = &loops[u.loop_idx];
        unsigned loop_tag = (u.pc ^ (u.pc >> 9)) & 0xFFFF;
        if (lp->tag == loop_tag && lp->conf >= 4 && lp->iter > 0) {
            if (lp->cur_iter + 1 >= lp->iter && final_tage) {
                final_tage = false;
            }
        }
        u.tage_pred = final_tage;
        // Compute perceptron prediction
        unsigned phash = (unsigned)path_hist ^ ((unsigned)(path_hist >> 19) & 0x7FF);
        u.perc_idx = (u.pc ^ (u.pc >> 20) ^ phash ^ ((unsigned)ghist & 0x7FF)) & (PERC_SIZE - 1);
        u.perc_sum = perc_weights[u.perc_idx][0];
        for (int i = 0; i < PERC_HIST; i++) {
            if ((ghist >> i) & 1) {
                u.perc_sum += perc_weights[u.perc_idx][i + 1];
            } else {
                u.perc_sum -= perc_weights[u.perc_idx][i + 1];
            }
        }
        u.perc_pred = (u.perc_sum >= 0);
        // Compute statistical corrector prediction
        u.sc_total = 0;
        for (int i = 0; i < SC_TABLES; i++) {
            unsigned int idx;
            switch(i) {
                case 0: idx = (u.pc ^ (ghist & 0x1FFFF)) & (SC_SIZE - 1); break;
                case 1: idx = ((u.pc >> 2) ^ (ghist >> 12) ^ (path_hist & 0x1FFF)) & (SC_SIZE - 1); break;
                case 2: idx = ((u.pc << 2) ^ (ghist >> 20) ^ (path_hist >> 7)) & (SC_SIZE - 1); break;
                default: idx = (u.pc ^ (ghist >> (i * 5)) ^ (path_hist >> (i * 4))) & (SC_SIZE - 1); break;
            }
            u.sc_indices[i] = idx;
            u.sc_sums[i] = sc[i][idx];
            int weight = (i == 0) ? 2 : ((i == 1) ? 3 : ((i == 2) ? 5 : 5));
            u.sc_total += sc[i][idx] * weight / 6;
        }
        // Compute local history prediction
        u.local_idx = (u.pc ^ (u.pc >> 6)) & (LOCAL_HIST_SIZE - 1);
        u.local_hist = local_hist[u.local_idx].hist;
        u.local_sum = local_pred[u.local_idx][u.local_hist];
        u.bias_idx = (u.pc ^ (ghist & 0xFFF)) & (BIAS_SIZE - 1);
        u.bias_sum = bias[u.bias_idx];
        // Combine neural components and use meta-predictor to select
        unsigned meta_hash = u.pc ^ (u.pc >> 20) ^ (ghist & 0x3FFFF) ^ ((unsigned)(path_hist >> 5) & 0x7FFF);
        u.meta_idx = meta_hash & (META_SIZE - 1);
        u.meta_ctr = meta[u.meta_idx];
        int neural_vote = u.perc_sum * 2 + u.sc_total + u.local_sum * 39;
        bool neural_pred = neural_vote >= 0;
        bool use_tage = (u.meta_ctr >= 2);
        // Override to neural if TAGE unstable and neural confident
        if (!u.tage_stable && u.hit_bank >= 0) {
            int abs_neural = (neural_vote >= 0) ? neural_vote : -neural_vote;
            int abs_local = (u.local_sum >= 0) ? u.local_sum : -u.local_sum;
            if (abs_local > 11 && abs_neural > 6) {
                use_tage = false;
            }
        }
        // Override to neural if provider weak and neural confident
        if (u.provider_weak) {
            int abs_neural = (neural_vote >= 0) ? neural_vote : -neural_vote;
            int threshold = (u.meta_ctr <= 1) ? 11 : 17;
            if (abs_neural > threshold) {
                use_tage = false;
            }
        }
        bool final_pred = use_tage ? u.tage_pred : neural_pred;
        u.direction_prediction(final_pred);
        return &u;
    }
    void update(branch_update *up, bool taken, unsigned int target) {
        if (!(bi.br_flags & BR_CONDITIONAL)) {
            ghist = (ghist << 1) | taken;
            path_hist = (path_hist << 2) ^ (bi.address >> 2);
            return;
        }
        my_update *mu = (my_update*)up;
        bool correct = (up->direction_prediction() == taken);
        branch_count++;
        // Periodically decay useful bits to allow replacement
        if ((branch_count & 0x1FFF) == 0) {
            for (int b = 0; b < TAGE_TABLES; b++) {
                for (int i = 0; i < TABLE_SIZE; i += 64) {
                    if (tage[b][i].u > 0) tage[b][i].u--;
                }
            }
        }
        // Update base predictor
        unsigned char &bc = base[mu->base_idx];
        if (taken) { if (bc < 3) bc++; } else { if (bc > 0) bc--; }
        // Update TAGE stability tracker
        bool tage_correct = (mu->tage_pred == taken);
        if (mu->hit_bank >= 0) {
            if (tage_correct) {
                if (stability[mu->stab_idx] < 3) stability[mu->stab_idx]++;
            } else {
                if (stability[mu->stab_idx] > 0) stability[mu->stab_idx]--;
            }
        }
        // Update TAGE provider entry and useful bits
        bool need_alloc = false;
        if (mu->hit_bank >= 0) {
            tage_entry *e = &tage[mu->hit_bank][mu->indices[mu->hit_bank]];
            unsigned ctr_max = (mu->hit_bank >= 2) ? 7 : 3;
            if (taken) {
                if (e->ctr < ctr_max) e->ctr++;
            } else {
                if (e->ctr > 0) e->ctr--;
            }
            bool p_ok = (mu->provider_pred == taken);
            bool a_ok = (mu->alt_pred == taken);
            if (p_ok != a_ok) {
                if (p_ok && e->u < 7) e->u++;
                else if (!p_ok && e->u > 0) e->u--;
            }
            need_alloc = (!p_ok && !a_ok);
        } else {
            bool base_pred = (base[mu->base_idx] >> 1) != 0;
            need_alloc = (base_pred != taken);
        }
        // Allocate new TAGE entries on misprediction
        if (need_alloc) {
            int allocated = 0;
            if (mu->hit_bank >= 0) {
                for (int b = mu->hit_bank + 1; b < TAGE_TABLES && allocated < 3; b++) {
                    tage_entry *e = &tage[b][mu->indices[b]];
                    if (e->u == 0) {
                        e->tag = mu->tags[b];
                        unsigned weak_val = (b >= 2) ? 4 : 2;
                        e->ctr = taken ? weak_val : (weak_val - 1);
                        e->u = 0;
                        allocated++;
                    }
                }
            } else {
                for (int b = TAGE_TABLES - 1; b >= 0 && allocated < 3; b--) {
                    tage_entry *e = &tage[b][mu->indices[b]];
                    if (e->u == 0) {
                        e->tag = mu->tags[b];
                        unsigned weak_val = (b >= 2) ? 4 : 2;
                        e->ctr = taken ? weak_val : (weak_val - 1);
                        e->u = 0;
                        allocated++;
                    }
                }
            }
        }
        // Train perceptron (if wrong or low confidence)
        bool perc_correct = (mu->perc_pred == taken);
        int abs_output = (mu->perc_sum >= 0) ? mu->perc_sum : -mu->perc_sum;
        if (!perc_correct || abs_output <= PERC_THETA) {
            signed char *weights = perc_weights[mu->perc_idx];
            if (taken) {
                if (weights[0] < 127) weights[0]++;
            } else {
                if (weights[0] > -128) weights[0]--;
            }
            for (int i = 0; i < PERC_HIST; i++) {
                bool hist_bit = ((ghist >> i) & 1);
                if (taken == hist_bit) {
                    if (weights[i + 1] < 127) weights[i + 1]++;
                } else {
                    if (weights[i + 1] > -128) weights[i + 1]--;
                }
            }
        }
        // Train meta-predictor when TAGE and neural disagree
        int neural_vote = mu->perc_sum * 2 + mu->sc_total + mu->local_sum * 39 + mu->bias_sum;
        bool neural_pred = neural_vote >= 0;
        bool neural_correct = (neural_pred == taken);
        if (neural_correct != tage_correct) {
            if (tage_correct) {
                if (meta[mu->meta_idx] < 3) meta[mu->meta_idx]++;
            } else {
                if (meta[mu->meta_idx] > 0) meta[mu->meta_idx]--;
            }
        }
        // Update loop predictor state
        loop_entry *lp = &loops[mu->loop_idx];
        unsigned loop_tag = (mu->pc ^ (mu->pc >> 9)) & 0xFFFF;
        if (lp->tag == loop_tag) {
            if (taken) {
                if (lp->cur_iter < 1023) lp->cur_iter++;
                if (lp->cur_iter >= lp->iter && lp->conf > 0) lp->conf--;
            } else {
                if (lp->cur_iter > 0) {
                    unsigned short new_iter = lp->cur_iter + 1;
                    if (new_iter > 1023) new_iter = 1023;
                    if (lp->iter == new_iter) {
                        if (lp->conf < 7) lp->conf++;
                    } else {
                        lp->iter = new_iter;
                        lp->conf = 0;
                    }
                }
                lp->cur_iter = 0;
            }
        } else {
            lp->tag = loop_tag;
            lp->conf = 0;
            lp->iter = 0;
            lp->cur_iter = taken ? 1 : 0;
        }
        // Train statistical corrector
        for (int i = 0; i < SC_TABLES; i++) {
            if (!correct || abs(mu->sc_sums[i]) < 22) {
                if (taken) {
                    if (sc[i][mu->sc_indices[i]] < 127) sc[i][mu->sc_indices[i]]++;
                } else {
                    if (sc[i][mu->sc_indices[i]] > -128) sc[i][mu->sc_indices[i]]--;
                }
            }
        }
        // Train local history predictor
        if (!correct || abs(mu->local_sum) < 4) {
            if (taken) {
                if (local_pred[mu->local_idx][mu->local_hist] < 127)
                    local_pred[mu->local_idx][mu->local_hist]++;
            } else {
                if (local_pred[mu->local_idx][mu->local_hist] > -128)
                    local_pred[mu->local_idx][mu->local_hist]--;
            }
        }
        local_hist[mu->local_idx].hist = ((mu->local_hist << 1) | taken) & ((1 << LOCAL_HIST_LEN) - 1);
        // Train bias weights
        if (!correct || abs(mu->bias_sum) < 10) {
            if (taken) {
                if (bias[mu->bias_idx] < 127) bias[mu->bias_idx]++;
            } else {
                if (bias[mu->bias_idx] > -128) bias[mu->bias_idx]--;
            }
        }
        // Shift outcome into global and path history
        ghist = (ghist << 1) | taken;
        path_hist = (path_hist << 2) ^ mu->pc ^ ((mu->pc >> 3) << 1);
    }
};




