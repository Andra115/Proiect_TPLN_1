# Evaluation Report

- **Model:** `models/byt5-diacritics/final`
- **Dataset:** `data/gold.jsonl` (360 examples)

## Overall and per-scenario CER / WER

| Scenario | n | CER | WER |
|---|---:|---:|---:|
| `cedilla` | 60 | 0.44% | 2.16% |
| `mixed` | 60 | 1.53% | 7.97% |
| `ocr` | 60 | 0.66% | 3.47% |
| `strip_all` | 60 | 1.00% | 5.44% |
| `strip_partial` | 60 | 0.85% | 4.60% |
| `wrong_diacritics` | 60 | 1.37% | 7.33% |
| `OVERALL` | 360 | 0.98% | 5.16% |

## Per-confusion restoration accuracy

How often the model produced the correct diacritic, given the input
had a degraded form. Counts only length-preserving examples
(62 skipped due to length mismatch).

| Confusion | Times degraded | Restored | Accuracy | Error rate |
|---|---:|---:|---:|---:|
| `I->Î` | 4 | 4 | 100.00% | 0.00% |
| `a->â` | 60 | 56 | 93.33% | 6.67% |
| `i->î` | 145 | 109 | 75.17% | 24.83% |
| `â->î` | 4 | 3 | 75.00% | 25.00% |
| `a->ă` | 232 | 199 | 85.78% | 14.22% |
| `S->Ș` | 2 | 0 | 0.00% | 100.00% |
| `Ş->Ș` | 1 | 0 | 0.00% | 100.00% |
| `s->ș` | 163 | 147 | 90.18% | 9.82% |
| `ş->ș` | 40 | 40 | 100.00% | 0.00% |
| `T->Ț` | 2 | 2 | 100.00% | 0.00% |
| `Ţ->Ț` | 1 | 1 | 100.00% | 0.00% |
| `t->ț` | 106 | 98 | 92.45% | 7.55% |
| `ţ->ț` | 25 | 25 | 100.00% | 0.00% |

## Detection precision / recall / F1

Treats each character position as a binary classification:
did the model correctly identify whether this position needed correction?

| Scenario | n | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `cedilla` | 59 | 67 | 20 | 0 | 77.01% | 100.00% | 87.01% |
| `mixed` | 34 | 413 | 20 | 14 | 95.38% | 96.72% | 96.05% |
| `ocr` | 32 | 9 | 20 | 0 | 31.03% | 100.00% | 47.37% |
| `strip_all` | 59 | 243 | 23 | 40 | 91.35% | 85.87% | 88.52% |
| `strip_partial` | 58 | 123 | 28 | 19 | 81.46% | 86.62% | 83.96% |
| `wrong_diacritics` | 56 | 740 | 29 | 32 | 96.23% | 95.85% | 96.04% |
| `OVERALL` | 298 | 1595 | 140 | 105 | 91.93% | 93.82% | 92.87% |

## Top error examples

The worst predictions by CER, useful for qualitative analysis.
Full list in `eval_worst_errors.jsonl`.

### 1. `wrong_diacritics` — CER 7.06%

- **Input:**     Koșecîki este un șat în comunî Liștvin dîn raionul Ovruci, regiunea Jițomir, Ucraina.
- **Target:**    Koșecikî este un sat în comuna Lîstvîn din raionul Ovruci, regiunea Jîtomîr, Ucraina.
- **Predicted:** Koseciki este un sat în comuna Listvin din raionul Ovruci, regiunea Jitomir, Ucraina.

### 2. `mixed` — CER 7.06%

- **Input:**     Koșecîki este un sat în comuna Listvîn din râîonul Ovruci, regîuneă Jițomir, Ucraina .
- **Target:**    Koșecikî este un sat în comuna Lîstvîn din raionul Ovruci, regiunea Jîtomîr, Ucraina.
- **Predicted:** Koseciki este un sat în comuna Listvin din raionul Ovruci, regiunea Jitomir, Ucraina.

### 3. `strip_all` — CER 5.88%

- **Input:**     Koseciki este un sat in comuna Listvin din raionul Ovruci, regiunea Jitomir, Ucraina.
- **Target:**    Koșecikî este un sat în comuna Lîstvîn din raionul Ovruci, regiunea Jîtomîr, Ucraina.
- **Predicted:** Koșeciki este un sat în comuna Listvin din raionul Ovruci, regiunea Jitomir, Ucraina.

### 4. `strip_partial` — CER 5.88%

- **Input:**     Koșeciki este un sat în comuna Listvîn din raionul Ovruci, regiunea Jîtomîr, Ucraina.
- **Target:**    Koșecikî este un sat în comuna Lîstvîn din raionul Ovruci, regiunea Jîtomîr, Ucraina.
- **Predicted:** Koșeciki este un sat în comuna Listvin din raionul Ovruci, regiunea Jitomar, Ucraina.

### 5. `cedilla` — CER 5.88%

- **Input:**     Koșecikî este un sat în comuna Lîstvîn din raionul Ovruci, regiunea Jîtomîr, Ucraina.
- **Target:**    Koșecikî este un sat în comuna Lîstvîn din raionul Ovruci, regiunea Jîtomîr, Ucraina.
- **Predicted:** Koșecika este un sat în comuna Listvin din raionul Ovruci, regiunea Jitomar, Ucraina.

### 6. `ocr` — CER 5.88%

- **Input:**     Koșecikî est e un sat în comuna Lîstvîn din raionul Ovruci, regiunea Jîtomîr , Ucraina.
- **Target:**    Koșecikî este un sat în comuna Lîstvîn din raionul Ovruci, regiunea Jîtomîr, Ucraina.
- **Predicted:** Koșeciki este un sat în comuna Listvin din raionul Ovruci, regiunea Jitomar, Ucraina.

### 7. `mixed` — CER 5.75%

- **Input:**     Sîtkivka este un șa ț in comuna Rozsîpne din răîonul Țroițke, regiunea Luhanșk, Ucraîna.
- **Target:**    Șatkivka este un sat în comuna Rozsîpne din raionul Troițke, regiunea Luhansk, Ucraina.
- **Predicted:** Sitkivka este un săt în comuna Rozsipne din raionul Troitke, regiunea Luhansk, Ucraina.

### 8. `strip_all` — CER 5.56%

- **Input:**     Meksunivka este un sat in comuna Nedanciici din raionul Ripki, regiunea Cernihiv, Ucraina.
- **Target:**    Mekșunivka este un sat în comuna Nedanciîci din raionul Ripkî, regiunea Cernihiv, Ucraina.
- **Predicted:** Meksunivka este un sat în comuna Nedănciici din raionul Rîpki, regiunea Cernihiv, Ucraina.

### 9. `strip_partial` — CER 5.56%

- **Input:**     Meksunivka este un sat in comuna Nedanciîci din raionul Ripki, regiunea Cernihiv, Ucraina.
- **Target:**    Mekșunivka este un sat în comuna Nedanciîci din raionul Ripkî, regiunea Cernihiv, Ucraina.
- **Predicted:** Meksunivka este un sat în comuna Nedănciici din raionul Rîpki, regiunea Cernihiv, Ucraina.

### 10. `wrong_diacritics` — CER 5.56%

- **Input:**     Mekșunîvka este un săt in comuna Nedanciîci din răionul Ripki, regiunea Cernîhîv, Ucrâină.
- **Target:**    Mekșunivka este un sat în comuna Nedanciîci din raionul Ripkî, regiunea Cernihiv, Ucraina.
- **Predicted:** Meksunivka este un sat în comuna Nedănciici din raionul Rîpki, regiunea Cernihiv, Ucraina.
