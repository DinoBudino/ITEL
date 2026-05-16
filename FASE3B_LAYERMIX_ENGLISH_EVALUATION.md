# FASE 3B: Evaluacion de V3/LayerMix en Ingles

**Objetivo:** Validar si LayerMix (Mejora A) mejora la transferencia multilingue frente al
modelo base de Fase 2.

**Checkpoint evaluado:** `research/checkpoints/v3_final/layermix/v3_layermix_fold8_of_10.pt`
(fold 8, bal_acc=0.9753, auc=0.9968, use_layer_mix=True)

**Fecha:** Mayo 2026

**Veredicto:** LayerMix **NO** mejora la transferencia al ingles. Empeora systematicamente
en todas las metricas.

---

## 1. Resultados Zero-Shot V3 LayerMix

| Metrica | Fase 2 Zero-Shot | V3 LayerMix Zero-Shot |
|---|---:|---:|
| AUC | 0.600 | 0.600 |
| Balanced Accuracy (raw, thr=0.20) | 0.500 | 0.500 |
| Balanced Accuracy (recalibrado) | **0.700** | 0.500 |
| Mejor threshold ingles | 0.9536 | 0.9999 |
| Scores Healthy (rango) | [0.58, 0.95] | [0.9999, 1.0000] |
| Scores Patients (rango) | [0.85, 0.98] | [1.0000, 1.0000] |

**Diagnostico:** El V3 LayerMix zero-shot produce **saturacion total**. Todos los scores
estan en ~1.0000, sin variacion entre sujetos. Esto es **peor** que Fase 2, donde al menos
habia gradacion (411 = 0.58, 607 = 0.85, 609 = 0.97). En Fase 2, la recalibracion de
umbral recuperaba bal_acc=0.700. En Fase 3, **ningun umbral puede separar** Healthy de
Patients porque todos los scores estan identicamente en el techo.

**Respuesta a preguntas clave:**
- `Subio el AUC zero-shot respecto a 0.600?` **NO** -- se mantiene en 0.600
- `Subio el best balanced accuracy respecto a 0.700?` **NO** -- bajo de 0.700 a 0.500

---

## 2. Linear Probing V3 LayerMix (LOSO)

| Metrica | Fase 2 Linear Probe LOSO | V3 LayerMix Linear Probe LOSO |
|---|---:|---:|
| Balanced Accuracy | **0.657** | 0.557 |
| AUC | **0.743** | 0.514 |
| Recall | **0.600** | 0.400 |
| Precision | 0.600 | 0.500 |
| F1 | **0.600** | 0.444 |

**Diagnostico:** El linear probing sobre context vectors V3 tambien es **peor** que Fase 2.
El AUC baja de 0.743 a 0.514 (-31%), indicando que los context vectors del modelo LayerMix
tienen **menos capacidad discriminativa** para ingles que los del modelo base.

**Respuesta a preguntas clave:**
- `Subio el linear probing respecto a AUC=0.743?` **NO** -- bajo a 0.514

---

## 3. Auditoria Geometrica V3

| Metrica | Fase 2 | V3 LayerMix |
|---|---:|---:|
| Geometry accuracy | **0.500** (6/12) | 0.417 (5/12) |
| EN Healthy -> CZ Healthy (mean dist) | 4.139 | 2.160 |
| EN Healthy -> CZ Patients (mean dist) | 1.649 | 2.129 |
| EN Patients -> CZ Healthy (mean dist) | 4.868 | 2.349 |
| EN Patients -> CZ Patients (mean dist) | 2.044 | 2.285 |

**Diagnostico:** Las distancias a centroides se comprimieron en V3. En Fase 2 habia una
separacion clara (Healthy EN mucho mas cerca de Patients CZ que de Healthy CZ, ratio ~2.5x).
En V3, Healthy y Patients en ingles estan practicamente a la misma distancia de ambos
centroides checos (~2.1-2.3). El espacio latente perdio la estructura que tenia en Fase 2.

**Sujeto 411:** Sigue siendo el unico Healthy geometricamente mas cerca del centroide
Healthy checo. Se mantiene como "puente" pero su score zero-shot ahora es 0.9999 (en
vez de 0.584 en Fase 2). El linear probing V3 le asigna 0.259 (correcto).

**Respuesta a preguntas clave:**
- `Mas sanos ingleses quedaron del lado del centroide sano checo?` **NO** -- geometria empeoro

---

## 4. Tabla Comparativa Final

| Metodo | AUC | Bal Acc | Recall | Precision | F1 |
|---|---:|---:|---:|---:|---:|
| Fase 2 Zero-Shot raw (thr=0.20) | 0.600 | 0.500 | 1.000 | 0.417 | 0.588 |
| Fase 2 Zero-Shot recalibrado (thr=0.95) | 0.600 | **0.700** | 0.400 | 1.000 | 0.571 |
| Fase 2 Linear Probe LOSO | **0.743** | **0.657** | **0.600** | **0.600** | **0.600** |
| V3 LayerMix Zero-Shot raw (thr=0.20) | 0.600 | 0.500 | 1.000 | 0.417 | 0.588 |
| V3 LayerMix Zero-Shot recalibrado | 0.600 | 0.500 | 0.000 | 0.000 | 0.000 |
| V3 LayerMix Linear Probe LOSO | 0.514 | 0.557 | 0.400 | 0.500 | 0.444 |

---

## 5. Per-Subject Detail V3

```
Subj  Label  ZS_Score  LP_Score  LP_OK  Geometry
 401    0     1.0000    0.4490    OK     patients (wrong)
 403    0     1.0000    0.3471    OK     patients (wrong)
 404    0     1.0000    0.6509    ERR    patients (wrong)
 405    0     1.0000    0.4199    OK     patients (wrong)
 409    0     1.0000    0.4657    OK     patients (wrong)
 410    0     1.0000    0.5706    ERR    patients (wrong)
 411    0     0.9999    0.2590    OK     healthy  (correct)  <- bridge
 574    1     1.0000    0.5990    OK     patients (correct)
 607    1     1.0000    0.3722    ERR    healthy  (wrong)
 609    1     1.0000    0.4290    ERR    patients (correct)
 611    1     1.0000    0.5581    OK     patients (correct)
 617    1     1.0000    0.3816    ERR    patients (correct)
```

---

## 6. Interpretacion

### Por que LayerMix empeora la transferencia

LayerMix hace que el modelo aprenda mejor los rasgos **especificos del checo** usando
informacion de las 12 capas del transformer. Esto es excelente para el rendimiento
intra-idioma (+6.9pp balanced accuracy en checo), pero **contraproducente para la
generalizacion cross-lingual**.

El fenomeno es analogo al overfitting de dominio: al tener acceso a 12 capas de features
en vez de 1, el clasificador aprende una representacion mas rica y detallada del espacio
acustico checo. Cuando se expone al ingles, esa riqueza se convierte en fragilidad: el
modelo encuentra **mas razones** para clasificar cualquier audio no-checo como anomalo
(Patients), y lo hace con mayor confianza (scores en 1.0 en vez de 0.85-0.98).

### Implicaciones para el proyecto

1. **LayerMix es una mejora valida para el modelo checo**, donde mejora todas las metricas.
2. **LayerMix NO es una mejora para transferencia multilingue**. Para ese proposito,
   el modelo base de Fase 2 (single-layer HuBERT) es mejor.
3. Si el objetivo final es un modelo multilingue, la estrategia correcta es:
   - mantener el backbone simple (single-layer o pocas capas)
   - invertir en **mas datos de entrenamiento en ingles**
   - o usar **tecnicas de domain adaptation** (adversarial, MMD, CORAL)

---

## 7. Artefactos Generados en Fase 3B

| Archivo | Descripcion |
|---|---|
| `research/processed_data/manifests/hubert_sequences_en_test_all_layers.csv` | Secuencias EN 12 capas |
| `research/scripts/evaluate_zero_shot_en_v3.py` | Evaluador zero-shot V3 |
| `research/scripts/linear_probe_en_v3.py` | Linear probing V3 LOSO |
| `research/scripts/audit_zero_shot_geometry_v3.py` | Auditoria geometrica V3 |
| `research/research_results/CSV/transfer_en/predictions_zero_shot_v3_layermix.csv` | Predicciones ZS V3 |
| `research/research_results/CSV/transfer_en/embeddings_zero_shot_v3_layermix.parquet` | Embeddings ZS V3 |
| `research/research_results/CSV/transfer_en/metrics_zero_shot_v3_layermix.json` | Metricas ZS V3 |
| `research/research_results/transfer_en/linear_probe_v3_layermix_report.json` | Reporte LP V3 |
| `research/research_results/transfer_en/diagnostic_audit_report_v3_layermix.json` | Auditoria geometrica V3 |
| `research/research_results/attention_viz/transfer_en/umap_linear_probe_v3_layermix.png` | UMAP LP V3 |
| `research/research_results/attention_viz/transfer_en/umap_zero_shot_cz_en_v3_layermix.png` | UMAP ZS V3 |
| `research/research_results/attention_viz/transfer_en/score_histogram_en_v3_layermix.png` | Histograma scores V3 |
| `FASE3B_LAYERMIX_ENGLISH_EVALUATION.md` | Este documento |

---

## 8. Recomendacion de Siguiente Paso

1. **No desplegar LayerMix para uso multilingue.**
2. Mantener el modelo V3 LayerMix como **modelo optimo para checo**.
3. Para la rama multilingue, usar el modelo base (single-layer) de Fase 2 con linear probing.
4. Evaluar Mejora B (embedding noise) para transferencia -- la regularizacion por ruido
   podria **ayudar** a la generalizacion cross-lingual, al contrario que LayerMix.
5. Considerar **fine-tuning ligero del modelo base** (no LayerMix) sobre datos ingleses
   si se adquieren mas muestras.
