# Precision, Recall, PR kriva i Average Precision — kompletno objašnjenje

Ovaj dokument prati implementaciju u [pointcept/utils/metric.py](../pointcept/utils/metric.py) i prateće vizualizacije u [viz_ap/](.).

---

## 1. Osnovne brojeće veličine

Za svaki binarni klasifikator (ili detektor pri fiksnom pragu) razlikujemo četiri kategorije:

| Skraćenica | Naziv | Šta znači |
|---|---|---|
| **TP** | True Positive | model je rekao "pozitivno" i bio je u pravu |
| **FP** | False Positive | model je rekao "pozitivno" ali je pogrešio (lažna uzbuna) |
| **FN** | False Negative | model je rekao "negativno" ali je propustio pozitivni primer |
| **TN** | True Negative | model je rekao "negativno" i bio je u pravu |

Kod **instance segmentation** posebno:
- TP = predikcija koja se IoU-om ≥ pragom poklapa sa **slobodnom** GT instancom (greedy matching)
- FP = predikcija koja nije pridružena nijednoj GT instanci
- FN = GT instanca koja nije pridružena nijednoj predikciji
- TN se obično ne broji (nema "negativnih instanci")

---

## 2. Precision i Recall

### Definicije

```
Precision = TP / (TP + FP)     ← od svih što sam rekao "da", koliko je tačno?
Recall    = TP / (TP + FN)     ← od svih stvarnih pozitiva, koliko sam uhvatio?
```

Za detektor sa **`num_gt`** GT instanci u skupu:
```
Recall = TP / num_gt
```
jer `TP + FN = num_gt` (svaka GT instanca je ili pogodjena (TP) ili propuštena (FN)).

### Intuicija

- **Visok Precision, nizak Recall**: konzervativan model — retko predviđa, ali kad predvidi, obično je u pravu. Propušta dosta GT-ova.
- **Visok Recall, nizak Precision**: agresivan model — predviđa puno, hvata sve GT-ove, ali ima i puno FP-ova.
- **Visok Precision i Recall**: idealan model.

### F1 score

Harmonijska sredina:
```
F1 = 2 · P · R / (P + R)
```
Visok F1 zahteva da su **i** Precision **i** Recall visoki. Maksimum F1 = 1.

Implementacija u [metric.py](../pointcept/utils/metric.py#L75-L90): `BestPRBase._get_best_results` traži threshold sa najvišim F1.

---

## 3. Kako se metrike menjaju sa threshold-om

Model obično daje **confidence score** za svaku predikciju. Threshold `t` određuje koje predikcije zadržavamo:
- predikcija je "pozitivna" akko `score ≥ t`

Vidi: [viz_ap/threshold_pr_curves.png](threshold_pr_curves.png) i [viz_ap/threshold_step_table.png](threshold_step_table.png).

### Šta se dešava kad spuštamo prag:

1. **Recall monotono raste** — svaki put kad smanjimo prag dodajemo neku predikciju; ako je ona TP, recall raste, ako je FP, recall stoji. Nikad ne pada.
2. **Precision oscilira** — pada na FP koracima, raste na TP koracima.
3. **F1** ima jasan maksimum negde u sredini (najbolji kompromis P-R).

### Primer iz vizualizacije (`num_gt=5`, 3 TP + 3 FP):

| t (prag) | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| → ∞ | 0 | 0 | 5 | 1.000 | 0.000 | 0.000 |
| 0.90 | 1 | 0 | 4 | 1.000 | 0.200 | 0.333 |
| 0.85 | 1 | 1 | 4 | 0.500 | 0.200 | 0.286 |
| 0.70 | 2 | 1 | 3 | 0.667 | 0.400 | 0.500 |
| **0.60** | **3** | **1** | **2** | **0.750** | **0.600** | **0.667** ← Best F1 |
| 0.40 | 3 | 2 | 2 | 0.600 | 0.600 | 0.600 |
| 0.30 | 3 | 3 | 2 | 0.500 | 0.600 | 0.545 |
| → 0 | 3 | 3 | 2 | 0.500 | 0.600 | 0.545 |

---

## 4. Precision-Recall kriva

Plot tačaka `(recall(t), precision(t))` za sve threshold-e `t`.

### Karakteristike:

- **X osa**: recall (od 0 do 1)
- **Y osa**: precision (od 0 do 1)
- **Smer obilaska**: idemo od najvišeg confidence-a ka najnižem (recall raste s leva na desno)
- **Oblik**: idealan model = pravougaonik (P=1, R=1); loš model = blizu dijagonale

### Kako se gradi (algoritam):

```python
order = np.argsort(-y_score, kind="stable")   # sort desc po score
y_true_sorted = y_true[order]

cum_tp = np.cumsum(y_true_sorted == 1)
cum_fp = np.cumsum(y_true_sorted == 0)

recall = cum_tp / num_gt
precision = cum_tp / (cum_tp + cum_fp)
```

To je upravo [_voc_ap u metric.py:410-417](../pointcept/utils/metric.py#L410-L417).

### Karakteristični izgled

PR krive su tipično **nazubljene** (zig-zag) — precision skače gore/dole kako prolazimo kroz mešavinu TP-ova i FP-ova. Zato se često koristi **interpolacija** (videti dole) pre računanja AP-a.

---

## 5. Average Precision (AP)

AP = **površina ispod PR krive**. Postoji više konvencija kako se ta površina računa.

### 5.1. Step-function AP (sklearn `average_precision_score`)

```
AP = Σ (R_n − R_{n-1}) · P_n
```

Suma "skokova recall-a" pomnoženih precision-om u toj tački. **Bez interpolacije.**

### 5.2. VOC 11-point interpolation (stari Pascal VOC)

Uzima se 11 fiksnih recall vrednosti `r ∈ {0, 0.1, 0.2, ..., 1.0}`. Za svaku se uzima maksimalna precision pri recall ≥ r. AP je njihov prosek:

```
AP = (1/11) · Σ_{r ∈ {0, 0.1, ..., 1.0}}  max_{r' ≥ r} P(r')
```

Manje precizno; deprecated.

### 5.3. VOC all-point interpolation (trenutno standard) — **ovo se koristi u kodu**

Vidi [_voc_ap u metric.py:398-428](../pointcept/utils/metric.py#L398-L428).

Algoritam:
1. Dodaj rubne tačke: `mrec = [0, R_1, R_2, ..., R_n, 1]`, `mpre = [0, P_1, ..., P_n, 0]`.
2. **Backwards-max** na precision (pravi monotono opadajuću krivu):
   ```python
   for i in range(len(mpre) - 2, -1, -1):
       mpre[i] = max(mpre[i], mpre[i + 1])
   ```
   Smisao: za svaku recall vrednost uzimamo **najveću precision desno** od nje. Ovo eliminiše "nazubljenost".
3. Sumiranje "skokova" recall-a:
   ```python
   ap = Σ (mrec[i+1] − mrec[i]) · mpre[i+1]
   ```
   gde se sumira samo gde je `mrec[i+1] != mrec[i]` (prave promene recall-a).

### 5.4. COCO AP @ [.5 : .05 : .95]

Glavna COCO metrika računa AP za **10 različitih IoU pragova** (0.50, 0.55, ..., 0.95) i uzima prosek:

```
AP_COCO = mean(AP@0.50, AP@0.55, AP@0.60, ..., AP@0.95)
```

Trenutni kod radi nešto slično (vidi [metric.py:461-465](../pointcept/utils/metric.py#L461-L465)):

```python
ap_values = [results[cls][f"ap_{int(ov*100)}"] for ov in self.overlaps if ov >= 0.5]
results[cls]["AP"] = float(np.mean(ap_values))
```

### 5.5. ScanNet konvencija

Identično (5.3) all-point interpolation, ali sa **eksplicitnom normalizacijom recall-a sa `num_gt`**:

```python
recall = cum_tp / float(num_gt)   # ne sa len(y_true)!
```

Razlog: u detection setupu, broj y_true entry-ja je broj predikcija, a ne broj GT instanci. Bez ove ispravke recall bi bio pogrešno normalizovan.

---

## 6. mAP — mean Average Precision

Računa se posebno za svaku klasu, pa se uzima prosek preko klasa:

```python
mAP = (1/C) · Σ_c AP_c
```

Vidi [metric.py:468-473](../pointcept/utils/metric.py#L468-L473):

```python
for mkey in metric_keys:
    values = [results[cls].get(mkey, 0.0) for cls in valid_classes]
    results[f"m{mkey}"] = float(np.mean(values))
```

Dobijamo:
- `mAP25` — mean AP pri IoU prag 0.25
- `mAP50` — mean AP pri IoU prag 0.50
- `mAP`   — mean AP@[.5:.95] (prosek pragova 0.5–0.95)

---

## 7. AP za instance segmentation (specifičnosti)

### 7.1. IoU između predikovane i GT maske

```python
IoU = |P ∩ G| / |P ∪ G|
```

U [metric.py:246-264](../pointcept/utils/metric.py#L246-L264), vektorizovano za sve parove:

```python
intersection = p_masks @ g_masks.T
union = p_sums + g_sums − intersection
iou = intersection / (union + 1e-6)
```

### 7.2. Greedy matching pri datom IoU pragu

Vidi [InstanceMatcher.match_at_threshold u metric.py:266-314](../pointcept/utils/metric.py#L266-L314). Algoritam:

1. Sortiraj predikcije po confidence-u silazno.
2. Za svaku predikciju, traži **najbolji slobodan GT** (najveći IoU iznad praga).
3. Ako pronađe match → `y_true = 1` (TP), GT se zaključa.
4. Ako ne → `y_true = 0` (FP).

**Bitno**: svaka GT instanca može biti dodeljena samo jednoj predikciji. Dupliranje predikcija kažnjava se kao FP.

Vizualizacija: [viz_ap/instance_matching.png](instance_matching.png).

### 7.3. Min region size filter

Predikcije i GT instance ispod `min_region_size` (npr. 100 tačaka) se izbacuju — smatraju se šumom. Vidi [metric.py:203, 237](../pointcept/utils/metric.py#L203).

### 7.4. Void / ignore regioni (ScanNet konvencija)

Predikcije koje padaju na ignore region (npr. ne-anotirane tačke) se **ne kažnjavaju kao FP**:
- broj tačaka predikcije se umanjuje za `void_intersection` ([metric.py:235](../pointcept/utils/metric.py#L235))
- u IoU-u, void preklapanje se oduzima od `p_sums` ([metric.py:260](../pointcept/utils/metric.py#L260))

---

## 8. Implicit vs. Explicit (−∞) FN — zašto je razlika

Vidi vizualizacije: [viz_ap/pr_curves_compare.png](pr_curves_compare.png), [viz_ap/ap_area_compare.png](ap_area_compare.png), [viz_ap/inf_decomposition.png](inf_decomposition.png).

### Pristup 1: implicitan FN (trenutni kod, ScanNet)

- `y_true` i `y_score` sadrže samo predikcije
- recall normalizovan sa `num_gt`
- PR kriva staje na `max_recall = TP/num_gt`
- AP integral iznad te tačke = 0

### Pristup 2: eksplicitan FN sa `score = −∞`

- Za svaki nepogodjen GT dodaje se `(y_true=1, y_score=−∞)`
- Nakon sortiranja, ti entry-ji su na kraju
- **Recall** se produžava do 1.0
- **Precision** raste u repu (cum_tp +1, cum_fp +0 sa svakim FN entry-jem)

### Numerička razlika (primer iz vizualizacije)

| Pristup | max recall | AP |
|---|---|---|
| 1 (implicit FN) | 0.6 | **0.500** |
| 2 (−∞ FN) | 1.0 | **0.750** |

Razlika +0.25 dolazi iz:
1. **Direktan doprinos** površini između recall ∈ (0.6, 1.0) sa precision = 0.625 → `0.4 · 0.625 = 0.25`
2. **Indirektan efekat** preko VOC backwards-max-a (može povući precision unazad i na nižim recallima)

### Šta je tačno (zašto je pristup 1 standard)

Model nikad nije dao predikcije za FN instance. PR kriva detektora **logički** ne može da reach recall > max_actual_recall. Pristup 2 fabrikuje tačke koje "pretendiraju" da je model nešto predviđao kad nije.

Konvencije koje koriste pristup 1:
- Pascal VOC
- ScanNet benchmark
- COCO AP

### Alternativna interpretacija — tačka `threshold → −∞`

Najčistiji način da se vidi šta `−∞` trik radi je da se pogleda **poslednja tačka sweep-a**, kad je threshold ispod svih score-ova (svi entry-ji su "predikcija"):

| Veličina | Sa −∞ FN trikom | Bez (pristup 1) |
|---|---|---|
| TP | `num_gt = 5` (originalna 3 + 2 "lažno detektovana") | 3 |
| FP | `num_fp = 3` | 3 |
| FN | **0** (svi GT-ovi su "pogodjeni" preko −∞) | 2 |
| Precision = `TP/(TP+FP)` | `5/8 = 0.625` | `3/6 = 0.500` |
| Recall = `TP/(TP+FN)` | `5/5 = 1.000` | `3/5 = 0.600` |

Drugim rečima, **−∞ trik svaki FN pretvara u jedan TP + jednu fiktivnu "predikciju"**:

> *"Dodajemo još jednu vrednost koja uzima sve instance kao da su tačno detektovane — uz cenu toga da u imeniocu kod precision-a imamo ukupan broj predikcija (uvećan za broj FN), a kod recall-a FN postaje 0."*

To je dvostruko "lažiranje":

1. **Recall**: FN nestaje iz imenioca → recall ide na 1.0 (model ništa nije predvideo za te instance!)
2. **Precision**: FN se broji **i** u brojiocu **i** u imeniocu (kao "predikcija") → precision raste ka `num_gt/(num_gt + num_fp)` umesto da padne ka 0

Kontrast sa pristupom 1: FN **ne postaje predikcija** — ostaje samo u imeniocu recall-a (`num_gt`). Max recall = `TP/num_gt`, a precision iznad te tačke je nedefinisan (VOC ga tretira kao 0).

Ova "threshold → −∞" tačka je **endpoint** PR krive u pristupu 2. Sve dodatne tačke koje −∞ trik uvodi (na recall = 0.8, 1.0, itd.) idu pravo ka tom endpoint-u, generišući lažnu površinu ispod krive koja AP pumpa naviše.

---

## 9. Klase u kodu

| Klasa | Opis | Linija |
|---|---|---|
| `BaseMetric` | apstraktni interface (update/compute/reset/sync) | [metric.py:18](../pointcept/utils/metric.py#L18) |
| `InstanceMatcher` | greedy matching + IoU computation | [metric.py:167](../pointcept/utils/metric.py#L167) |
| `InstanceAveragePrecision` | AP@[.25,.5,...,.95] po klasi + mAP | [metric.py:340](../pointcept/utils/metric.py#L340) |
| `MatchedOnlyInstanceMeanIoU` | mIoU samo nad TP parovima | [metric.py:481](../pointcept/utils/metric.py#L481) |
| `InstanceMeanIoU` | mIoU sa imeniteljem = num_gt (kažnjava FN) | [metric.py:584](../pointcept/utils/metric.py#L584) |
| `GTInstanceIoU` | best-IoU per GT (oracle pogled) | [metric.py:688](../pointcept/utils/metric.py#L688) |
| `BestF1Bundle` | binary AP + best-F1 threshold (pixel-level) | [metric.py:120](../pointcept/utils/metric.py#L120) |

### Pipeline `InstanceAveragePrecision`:

1. **`update(pred, gt)`** ([metric.py:373](../pointcept/utils/metric.py#L373)):
   - `matcher.assign(...)` — popuni IoU matrice
   - za svaki IoU prag → greedy match → akumuliraj y_true/y_score po klasi i pragu
   - akumuliraj `num_gt` po klasi (nezavisno od praga)

2. **`sync()`** ([metric.py:430](../pointcept/utils/metric.py#L430)) — DDP all-gather

3. **`compute()`** ([metric.py:449](../pointcept/utils/metric.py#L449)):
   - za svaku klasu × svaki prag → `_voc_ap(...)` → `ap_25, ap_50, ..., ap_95`
   - `AP25 = ap_25`, `AP50 = ap_50`, `AP = mean(ap_50,...,ap_95)`
   - mAP = prosek preko valid klasa

---

## 10. Kratka tabela: koja metrika znači šta

| Metrika | Šta meri | Kad da je gledaš |
|---|---|---|
| **AP25** | AP pri labavom IoU pragu (0.25) | grub overlap dovoljan; lako |
| **AP50** | AP pri standardnom pragu (0.50) | klasična detection metrika |
| **AP** = AP@[.5:.95] | strogi prosek pragova 0.50–0.95 | meri preciznost segmentacije |
| **mAP25/50** | prosek AP-a preko klasa | balansirano izmedju klasa |
| **F1@best** | maksimalan F1 preko threshold-a | nezavisno od kalibracije confidence-a |
| **Best Precision/Recall** | P i R pri istom thresholdu kao best F1 | operating point za inference |
| **mIoU** | prosečno IoU preko klasa | segmentation kvalitet (ne detection) |

---

## 11. Reference

### Kod
- [pointcept/utils/metric.py](../pointcept/utils/metric.py) — sva implementacija
- [pointcept/utils/metric.py:398-428](../pointcept/utils/metric.py#L398-L428) — `_voc_ap`
- [pointcept/utils/metric.py:266-314](../pointcept/utils/metric.py#L266-L314) — greedy matching

### Vizualizacije
- [viz_ap/instance_matching.png](instance_matching.png) — TP/FP/FN matching
- [viz_ap/threshold_pr_curves.png](threshold_pr_curves.png) — P/R/F1 vs threshold
- [viz_ap/threshold_step_table.png](threshold_step_table.png) — tabela po koracima
- [viz_ap/pr_curves_compare.png](pr_curves_compare.png) — PR krive (oba pristupa)
- [viz_ap/ap_area_compare.png](ap_area_compare.png) — AP površina dekompozicija
- [viz_ap/inf_decomposition.png](inf_decomposition.png) — efekat −∞ FN po koracima

### Skripte koje generišu slike
- [viz_ap/make_figures.py](make_figures.py)
- [viz_ap/make_threshold_figures.py](make_threshold_figures.py)
- [viz_ap/make_decomposition_figure.py](make_decomposition_figure.py)

### Literatura
- Pascal VOC 2012 dev kit (originalna VOC AP definicija)
- COCO Detection Evaluation (Lin et al.) — AP@[.5:.95]
- ScanNet benchmark (Dai et al.) — instance AP konvencija
