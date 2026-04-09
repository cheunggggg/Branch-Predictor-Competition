# Hybrid Branch Predictor 

A hybrid branch predictor for the UCLA ECE M116C branch prediction competition (Fall 2025), based on the CBP-2 framework. Combines TAGE, perceptron, statistical corrector, local history, loop detection, and a meta-predictor into a single ensemble, targeting minimum MPKI on the CBP-2 SPEC trace suite.

---

## Architecture Overview

The predictor is a layered ensemble. At the top level, a **meta-predictor** chooses between a TAGE prediction and a **neural ensemble** prediction. Two override rules can force the neural path even when the meta-predictor favors TAGE.

```
                    ┌──────────────────────────────────┐
                    │           Meta-Predictor          │
                    │  (4M-entry 2-bit saturating ctr)  │
                    └────────────┬─────────────┬────────┘
                                 │             │
                           TAGE path     Neural path
                                 │             │
                    ┌────────────▼──┐   ┌──────▼────────────────────────┐
                    │  TAGE (5 bnk) │   │ Perceptron × 2                │
                    │  + Loop pred  │   │ + Statistical Corrector × 1   │
                    │  + Stability  │   │ + Local History × 39          │
                    └───────────────┘   │ + Bias                        │
                                        └───────────────────────────────┘

Override to neural if:
  (a) TAGE unstable AND |local_sum| > 11 AND |neural_vote| > 6
  (b) TAGE provider weak AND |neural_vote| > threshold (11 or 17)
```

---

## Components

### 1. TAGE (Tagged Geometric History Length Predictor)

5 banks with geometric history lengths and per-bank tag widths:

| Bank | History Length | Tag Bits | Counter Width |
|------|---------------|----------|---------------|
| 0 | 0 (base) | 7 | 2-bit (0–3) |
| 1 | 1 | 9 | 2-bit (0–3) |
| 2 | 8 | 11 | 3-bit (0–7) |
| 3 | 30 | 12 | 3-bit (0–7) |
| 4 | 67 | 13 | 3-bit (0–7) |

Each bank has 131,072 entries. Tags are computed by folding global history, a scrambled version of global history, and path history, XORed with the PC. Indices are computed with a five-way hash mixing PC, folded global history, path history, and a bank-specific constant.

**Alt-prediction:** If the provider's counter is in a weak state and its useful bit is 0, the prediction falls back to the next matching bank (alt-bank) or the base predictor.

**Useful bit decay:** Every 8,192 branches, useful bits across all banks are decremented by 1 to allow stale entries to be replaced.

**Allocation:** On misprediction, up to 3 new entries with zero useful bits are allocated in banks with higher history than the provider. New counters are initialized to a weak taken/not-taken value.

### 2. Stability Tracker

A 16,384-entry 2-bit saturating counter array indexed by PC. Tracks whether TAGE has been consistently correct for a given PC. Used to trigger the neural override when TAGE is unreliable.

### 3. Loop Predictor

16,384 entries, each tracking: tag, expected iteration count, current iteration count, and a 3-bit confidence counter. When a loop entry is confident (conf ≥ 4) and the current iteration is about to reach the expected count, the predictor overrides TAGE to predict not-taken (loop exit).

Iteration counts are capped at **1,023** (`MAX_ITER`).

### 4. Perceptron

4,194,304 weight vectors, each with **65 weights** (1 bias + 64 history weights, `PERC_HIST = 64`). Weights are 8-bit signed integers (−128 to 127).

Index is hashed from PC, path history, and the lower 11 bits of global history. The dot product of history bits (±1) and weights gives a signed output. Training occurs when the prediction is wrong **or** when `|output| ≤ 109` (`PERC_THETA`).

### 5. Statistical Corrector (SC)

4 tables of 131,072 signed 8-bit counters each, indexed by different hashes of PC, global history, and path history:

| Table | Index Hash |
|-------|-----------|
| 0 | `PC ^ (ghist & 0x1FFFF)` |
| 1 | `(PC >> 2) ^ (ghist >> 12) ^ (path_hist & 0x1FFF)` |
| 2 | `(PC << 2) ^ (ghist >> 20) ^ (path_hist >> 7)` |
| 3 | `PC ^ (ghist >> 20) ^ (path_hist >> 16)` |

Contributions are weighted (2, 3, 5, 5) and scaled by 1/6 before summing. Training occurs when wrong **or** when `|sum| < 22`.

### 6. Local History Predictor

32,768 local history registers of 17-bit width (`LOCAL_HIST_LEN = 17`). Each PC is hashed to a history register, which indexes into a 32768 × 131072 table of signed 8-bit counters. Training occurs when wrong **or** when `|sum| < 4`.

### 7. Bias

131,072 signed 8-bit weights indexed by `PC ^ (ghist & 0xFFF)`. Provides a PC-specific directional bias. Training occurs when wrong **or** when `|sum| < 10`.

### 8. Neural Ensemble Vote

The four neural components are combined with fixed weights:

```
neural_vote = perc_sum × 2 + sc_total + local_sum × 39 + bias_sum
```

### 9. Meta-Predictor

4,194,304 2-bit saturating counters indexed by a hash of PC, global history, and path history. Increments when TAGE is correct and neural is wrong; decrements otherwise. A value ≥ 2 favors TAGE.

---

## Storage Overhead

| Component | Size |
|-----------|------|
| TAGE (5 banks × 131072 × 4 B) | ~2.5 MB |
| Loop predictor (16384 × 7 B) | ~0.11 MB |
| Perceptron weights (4194304 × 65 B) | ~260 MB |
| Statistical corrector (4 × 131072 B) | ~0.5 MB |
| Local history registers (32768 × 4 B) | ~0.13 MB |
| Local predictor table (32768 × 131072 B) | ~4 GB |
| Bias (131072 B) | ~0.13 MB |
| Meta-predictor (4194304 B) | ~4 MB |
| Stability tracker (16384 B) | ~16 KB |
| Base predictor (131072 B) | ~128 KB |

> **Note:** The local predictor table (4 GB) and perceptron weight array (260 MB) are not realistic for hardware implementation. These are sized for maximum simulation accuracy in the competition context.

---

## Files

```
my_predictor.c   — Full predictor implementation (drop-in replacement for my_predictor.h)
```

The CBP-2 infrastructure expects the predictor in `src/my_predictor.h`. Rename or include accordingly.

---

## Building and Running

```bash
# From the src/ directory
make

# From the top-level cbp2/ directory
csh run_traces
```

Output is one line per trace showing the benchmark name and its MPKI, followed by an average:

```
traces/164.gzip/gzip.trace.bz2   <mpki>
...
average MPKI: <value>
```

---

## Key Design Decisions

**TAGE as the primary predictor.** TAGE with geometric history lengths naturally captures branch behavior across a wide range of history depths. The 5-bank configuration with histories [0, 1, 8, 30, 67] covers short and medium patterns while the base table handles branches with no useful history correlation.

**Neural ensemble as a fallback.** Perceptron excels on branches with strong history correlations that TAGE misses due to aliasing. The statistical corrector and local history add complementary signal. The `× 39` weight on local history reflects its high discriminative power for loop-like and call-site-specific branches.

**Meta-predictor arbitration.** Rather than always deferring to one component, a learned 2-bit counter per PC-history context tracks which component has been more reliable.

**Stability-aware override.** A dedicated stability tracker detects when TAGE has recently been mispredicting at a given PC and allows the neural ensemble to take over, even when the meta-predictor would normally favor TAGE.

**Loop predictor.** Fixed-iteration loops produce a highly predictable pattern (taken N−1 times, then not-taken once) that neither TAGE nor perceptron handles efficiently. The loop predictor handles this case directly once it has learned the iteration count with sufficient confidence.
