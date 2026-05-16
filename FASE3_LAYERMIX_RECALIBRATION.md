# FASE 3F: Recalibracion de Precision para V3 LayerMix

**Objetivo:** Mejorar la precision de `V3 LayerMix` sin cambiar arquitectura, solo ajustando el umbral de decision.

**Fecha:** Mayo 2026

**Resultado:** Exito completo. La precision sube drasticamente solo con recalibrar el threshold de `0.20` a `0.61`, manteniendo recall clinico de `1.0`.

---

## 1. Resultados principales

### Threshold actual vs recalibrado

| Metrica | Original (thr=0.20) | Recalibrado (thr=0.61) | Delta |
|---|---:|---:|---:|
| Sample Precision | 0.8227 | **0.8885** | **+0.0658** |
| Sample Recall | 0.9965 | 0.9555 | -0.0410 |
| Sample Balanced Accuracy | 0.8533 | **0.8968** | **+0.0435** |
| Sample F1 | 0.9013 | **0.9208** | **+0.0195** |
| **Subject Precision** | 0.7397 | **0.9310** | **+0.1913** |
| **Subject Recall** | 1.0000 | **1.0000** | 0.0000 |
| **Subject Balanced Accuracy** | 0.7841 | **0.9545** | **+0.1704** |
| **Subject F1** | 0.8504 | **0.9643** | **+0.1139** |

### Interpretacion

El threshold original (`thr=0.20`) fue elegido por `train_optimized_v3.py` priorizando
exclusivamente recall. Eso empujaba a predecir casi todo como `Patients`, resultando en:

- recall perfecto
- precision castigada
- muchos falsos positivos

Al subir el threshold a `thr=0.61`, el modelo mantiene el recall clínico a nivel sujeto
(`1.00`) y mejora drásticamente la precisión (`+19.1pp`). Esto elimina una gran cantidad
de falsos positivos sin perder sensibilidad para los pacientes reales.

---

## 2. Comparacion de 4 politicas de threshold

| Politica | Threshold | Sample Prec | Sample Rec | Subj Prec | Subj Rec | Subj F1 |
|---|---:|---:|---:|---:|---:|---:|
| Current (recall-max) | 0.20 | 0.8227 | 0.9965 | 0.7397 | 1.0000 | 0.8504 |
| Best Balanced Accuracy | 0.62 | 0.8929 | 0.9550 | 0.9310 | 1.0000 | 0.9643 |
| Best F1 | 0.58 | 0.8776 | 0.9661 | 0.9310 | 1.0000 | 0.9643 |
| **Best Prec (rec >= 0.95)** | **0.61** | **0.8885** | 0.9555 | **0.9310** | 1.0000 | **0.9643** |

Las politicas "Best Balanced Accuracy", "Best F1" y "Best Precision constrained" son muy
similares (umbrales 0.58-0.62) y todas superan ampliamente al criterio original.

---

## 3. Sujetos criticos (H26-H29)

| Sujeto | Mean Score | FP actual (thr=0.20) | FP recalibrado (thr=0.61) | Mejora? |
|---|---:|---:|---:|---|
| H26 | 0.9905 | 24/24 | 24/24 | No -- 100% FP |
| H27 | 0.8933 | 25/27 | 25/27 | No |
| H28 | 0.8991 | 31/33 | 31/33 | No |
| H29 | 0.9822 | 27/27 | 27/27 | No |

**Diagnostico:** H26-H29 tienen scores muy altos (0.89-0.99), por encima incluso del
nuevo threshold (0.61). Estos sujetos requieren mejora arquitectonica o de entrenamiento,
no de calibracion de umbral. Son semanticamente "Patients" para el modelo.

---

## 4. Conclusion

**Si se puede mejorar la precision de V3 LayerMix**, y de forma muy significativa,
sin cambiar la arquitectura.

El problema original no era arquitectonico, sino de **calibracion de decision**:
el criterio de seleccion de checkpoint por `recall` puro y el threshold fijo de `0.20`
estaban sesgando el modelo hacia predicciones de `Patients` en exceso.

### Recomendacion

1. **Adoptar `thr=0.61` como umbral clinico estandar para V3 LayerMix**
2. Mantener el mismo modelo sin reentrenar
3. La metrica clave a reportar del modelo oficial checo deberia ser:
   - `subject balanced_accuracy = 0.9545`
   - `subject precision = 0.9310`
   - `subject recall = 1.0000`
4. Los sujetos H26-H29 requieren atencion clinica/arquitectonica adicional

### Respuesta a la pregunta central

> El problema de precision de V3 LayerMix era principalmente de **calibracion de decision**, no arquitectonico.

---

## 5. Artefactos generados

| Archivo | Descripcion |
|---|---|
| `research/scripts/recalibrate_layermix.py` | Script de recalibracion |
| `research/research_results/CSV/threshold_sweep_v3_layermix.csv` | Barrido fino 0.01 (95 thresholds) |
| `research/research_results/CSV/subject_level_threshold_tradeoff_v3_layermix.csv` | Tradeoff a nivel sujeto |
| `research/research_results/CSV/v3_layermix_recalibration_comparison.csv` | Comparacion de 4 politicas |
| `research/research_results/CSV/v3_layermix_critical_subjects_recalibration.csv` | Sujetos H26-H29 |
| `research/research_results/CSV/metrics_v3_layermix_recalibrated.json` | Metricas finales recalibradas |
| `research/research_results/attention_viz/final_comparison/v3_layermix_threshold_tradeoff.png` | Curva trade-off |
| `research/research_results/attention_viz/final_comparison/v3_layermix_precision_recall_curve.png` | Curva P-R sujeto |
| `FASE3_LAYERMIX_RECALIBRATION.md` | Este documento |
