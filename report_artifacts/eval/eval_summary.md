# Evaluation Report

- **Model:** `models/byt5-diacritics/final`
- **Dataset:** `data/val.jsonl` (3000 examples)

## Overall and per-scenario CER / WER

| Scenario | n | CER | WER |
|---|---:|---:|---:|
| `cedilla` | 500 | 0.06% | 0.31% |
| `mixed` | 500 | 0.61% | 3.11% |
| `ocr` | 500 | 0.15% | 0.88% |
| `strip_all` | 500 | 0.39% | 2.09% |
| `strip_partial` | 500 | 0.21% | 1.19% |
| `wrong_diacritics` | 500 | 0.50% | 2.67% |
| `OVERALL` | 3000 | 0.32% | 1.70% |

## Per-confusion restoration accuracy

How often the model produced the correct diacritic, given the input
had a degraded form. Counts only length-preserving examples
(520 skipped due to length mismatch).

| Confusion | Times degraded | Restored | Accuracy | Error rate |
|---|---:|---:|---:|---:|
| `I->Î` | 114 | 106 | 92.98% | 7.02% |
| `a->â` | 495 | 467 | 94.34% | 5.66% |
| `î->â` | 8 | 6 | 75.00% | 25.00% |
| `i->î` | 1269 | 1235 | 97.32% | 2.68% |
| `â->î` | 53 | 52 | 98.11% | 1.89% |
| `a->ă` | 3650 | 3506 | 96.05% | 3.95% |
| `S->Ș` | 36 | 31 | 86.11% | 13.89% |
| `Ş->Ș` | 8 | 8 | 100.00% | 0.00% |
| `s->ș` | 1175 | 1140 | 97.02% | 2.98% |
| `ş->ș` | 340 | 340 | 100.00% | 0.00% |
| `T->Ț` | 10 | 3 | 30.00% | 70.00% |
| `Ţ->Ț` | 3 | 3 | 100.00% | 0.00% |
| `t->ț` | 1859 | 1805 | 97.10% | 2.90% |
| `ţ->ț` | 567 | 567 | 100.00% | 0.00% |

## Detection precision / recall / F1

Treats each character position as a binary classification:
did the model correctly identify whether this position needed correction?

| Scenario | n | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `cedilla` | 497 | 918 | 25 | 0 | 97.35% | 100.00% | 98.66% |
| `mixed` | 256 | 4165 | 36 | 27 | 99.14% | 99.36% | 99.25% |
| `ocr` | 248 | 171 | 28 | 0 | 85.93% | 100.00% | 92.43% |
| `strip_all` | 495 | 3367 | 92 | 172 | 97.34% | 95.14% | 96.23% |
| `strip_partial` | 496 | 1677 | 87 | 42 | 95.07% | 97.56% | 96.30% |
| `wrong_diacritics` | 488 | 9460 | 97 | 98 | 98.99% | 98.97% | 98.98% |
| `OVERALL` | 2480 | 19758 | 365 | 339 | 98.19% | 98.31% | 98.25% |

## Top error examples

The worst predictions by CER, useful for qualitative analysis.
Full list in `eval_worst_errors.jsonl`.

### 1. `mixed` — CER 29.17%

- **Input:**     Daniel Drâgomir este de neg asiț sî va fî dât in urmărirePolâtia Capîtalei l-a cautâț dupa ce a fost condamnât definitîv, insă nu l-a gașițFoștul ofiter SRI â fost condămnat definițiv la 3 ani si 10 luni de inchișoare
- **Target:**    Daniel Dragomir este de negăsit și va fi dat în urmărirePoliția Capitalei l-a căutat după ce a fost condamnat definitiv, însă nu l-a găsitFostul ofițer SRI a fost condamnat definitiv la 3 ani și 10 luni de închisoare
- **Predicted:** Daniel Dragomir este de negăsit și va fi dat în urmărirePoliția Capitalei l-a căutat după ce a fost condamnat definitiv la 3 ani și 10 luni de închisoare

### 2. `strip_all` — CER 13.95%

- **Input:**     Nu doar ca stie sa pasca vacile si sa le mane la adapatoare, dar poate sa si le mulga.
- **Target:**    Nu doar că știe să pască vacile și să le mâne la adăpătoare, dar poate să și le mulgă.
- **Predicted:** Nu doar ca stie sa pasca vacile si sa le mane la adapatoare, dar poate sa si le mulga.

### 3. `wrong_diacritics` — CER 9.96%

- **Input:**     Cu ocîzia implinîrii a 30 de ani de la âderarea Romaniei lă Orgănîzația Internationala a Frâncofonîei (OIF), Însțituțul Culțural Romăn de la Madrid, in colăborare cu Însțitutul Francez din Madrid, cu sprîjînul Ambîșîdei Romanieî in Regatul Spaniei si al Conșulățului General al Romaniei lă Șevîlla, orgânizeăza concerțele de jazz „Drîve” șusținute de trio-ul tîmisorean JazzyBIT, vineri, 3 marțîe, la Teațrul Instițuțului Francez din Madrid si sambăta, 4 marțîe, la Muzeul Interactiv al Muzicîi MIMMA din Málaga.
- **Target:**    Cu ocazia împlinirii a 30 de ani de la aderarea României la Organizația Internațională a Francofoniei (OIF), Institutul Cultural Român de la Madrid, în colaborare cu Institutul Francez din Madrid, cu sprijinul Ambasadei României în Regatul Spaniei și al Consulatului General al României la Sevilla, organizează concertele de jazz „Drive” susținute de trio-ul timișorean JazzyBIT, vineri, 3 martie, la Teatrul Institutului Francez din Madrid și sâmbătă, 4 martie, la Muzeul Interactiv al Muzicii MIMMA din Málaga.
- **Predicted:** Cu ocazia împlinirii a 30 de ani de la aderarea României la Organizația Internațională a Francofoniei (OIF), Institutul Cultural Român de la Madrid, în colaborare cu Institutul Francez din Madrid, cu sprijinul Ambasadei României în Regatul Spaniei și al Consulatului General al României la Sevilla, organizează concertele de jazz „Drave” susținute de trio-ul timișorean JazzyBIT, vineri, 3 martie, la Teatrul Institutului Francez din Madrid și sâmbătă, 4 martie, 

### 4. `wrong_diacritics` — CER 9.09%

- **Input:**     Lî dața de 11 iunie a.
- **Target:**    La data de 11 iunie a.
- **Predicted:** La data de 11 iunie.

### 5. `mixed` — CER 9.09%

- **Input:**     Lî data de 11 iunie î.
- **Target:**    La data de 11 iunie a.
- **Predicted:** La data de 11 iunie.

### 6. `strip_all` — CER 8.33%

- **Input:**     Tantarii prefera sangele cu grupa 0.
- **Target:**    Țânțarii preferă sângele cu grupa 0.
- **Predicted:** Tântării preferă sângele cu grupa 0.

### 7. `strip_partial` — CER 8.33%

- **Input:**     Țântarii preferă sangele cu grupa 0.
- **Target:**    Țânțarii preferă sângele cu grupa 0.
- **Predicted:** Tântării preferă sângele cu grupa 0.

### 8. `wrong_diacritics` — CER 8.33%

- **Input:**     Tanțărâi prefera sângele cu grupa 0.
- **Target:**    Țânțarii preferă sângele cu grupa 0.
- **Predicted:** Tânării preferă sângele cu grupa 0.

### 9. `ocr` — CER 8.33%

- **Input:**     Țânțarii preferă sângele cu grupa 0.
- **Target:**    Țânțarii preferă sângele cu grupa 0.
- **Predicted:** Tântării preferă sângele cu grupa 0.

### 10. `mixed` — CER 8.33%

- **Input:**     Țantîrii prefera săngele cu grupâ 0.
- **Target:**    Țânțarii preferă sângele cu grupa 0.
- **Predicted:** Tântării preferă sângele cu grupa 0.
