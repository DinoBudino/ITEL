# FASE 3F: Post-Calibracion de V3 LayerMix

**Objetivo:** Mejorar precision de `V3 LayerMix` con calibracion probabilistica, thresholds
por subtarea y analisis de hard negatives, sin tocar arquitectura.

**Fecha:** Mayo 2026

**Resultado:** La recalibracion por threshold (fase 3F-A) ya capturo todo el beneficio.
Ni Platt, ni temperature scaling, ni isotonic, ni thresholds por subtarea aportan mejora
adicional significativa sobre `thr=0.61`. El limite actual es estructural (hard negatives).

---

## 1. Resultados de Calibracion Probabilistica

| Estrategia | Precision | Recall | F1 | Bal Acc | Brier |
|---|---:|---:|---:|---:|---:|
| Original (thr=0.20) | 0.7397 | 1.0000 | 0.8504 | 0.7841 | 0.0604 |
| Recalibrado (thr=0.61) | 0.9310 | 1.0000 | 0.9643 | 0.9545 | 0.0604 |
| Platt Scaling | 0.9310 | 1.0000 | 0.9643 | 0.9545 | 0.0566 |
| Temperature Scaling | 0.9310 | 1.0000 | 0.9643 | 0.9545 | 0.0520 |
| Isotonic Regression | 0.9310 | 1.0000 | 0.9643 | 0.9545 | **0.0313** |
| Sub-task thresholds | 0.9310 | 1.0000 | 0.9643 | 0.9545 | N/A |

### Interpretacion

- **Ninguna tecnica de calibracion mejoro las metricas de decision** (precision, recall,
  F1, balanced accuracy). El cambio de threshold de `0.20` a `0.61` ya habia capturado
  toda la mejora posible via decision threshold.

- **Isotonic regression si mejoro la calibracion probabilistica** (Brier score baja de
  0.0604 a 0.0313, una reduccion del 48%). Esto significa que los scores calibrados por
  isotonic son mas fiables como probabilidades, aunque no cambien la decision binaria.

- **Temperature scaling** encontro `best_temp = 0.2`, es decir, los scores raw del modelo
  ya estan muy bien calibrados y necesitan muy poca correccion de temperatura.

---

## 2. Decision por Subtarea

Los thresholds por grupo silabico (`syllable_group`) **no aportaron mejora**. El motivo
principal es que la distribucion de datos por grupo esta muy sesgada (word_1syll y
consonant dominan), y los thresholds optimos por grupo no difieren significativamente
del threshold global de `0.61`.

---

## 3. Analisis de Hard Negatives

| Sujeto | Mean Score | FP Rate (thr=0.61) |
|---|---:|---:|
| H26 | 0.9907 | 100% |
| H27 | 0.8916 | 93.5% |
| H28 | 0.8959 | 94.0% |
| H29 | 0.9818 | 100% |

Los cuatro sujetos criticos tienen scores medios muy por encima de `0.61`. **Ninguna
tecnica de calibracion puede corregir estos FP**, porque el modelo les asigna scores
extremadamente altos de forma consistente. Esto confirma que:

- No es un problema de calibracion
- Es un problema estructural: el modelo "ve TEL" en estos sanos

---

## 4. Comparacion Final

| Estrategia | Precision | Recall | F1 | Bal Acc | Recomendacion |
|---|---:|---:|---:|---:|---|
| Original (thr=0.20) | 0.74 | 1.00 | 0.85 | 0.78 | Descartado |
| Recalibrado (thr=0.61) | 0.93 | 1.00 | 0.96 | 0.95 | **GO - Estandar** |
| Isotonic | 0.93 | 1.00 | 0.96 | 0.95 | Opcional (mejor Brier) |
| Sub-task thresholds | 0.93 | 1.00 | 0.96 | 0.95 | No necesario |

---

## 5. Conclusion

**La mejora de precision de V3 LayerMix se logro completamente con la recalibracion**
**de threshold (thr=0.61). Las tecnicas adicionales de calibracion probabilistica no**
**aportan mejora en las metricas de decision binaria.**

### Recomendacion

1. **Adoptar `thr=0.61` como umbral clinico estandar para V3 LayerMix**
2. **Opcionalmente aplicar isotonic regression** para obtener probabilidades mejor
   calibradas (menor Brier score), util para reporting clinico de confianza
3. **Los sujetos H26-H29 requieren intervencion estructural** (reentrenamiento con
   hard-negative mining, focal loss, o analisis clinico detallado de estos casos)

### Respuesta a la pregunta central

> El limite actual de precision de V3 LayerMix ya es estructural, no de calibracion.
> El modelo alcanza `subject precision = 0.9310` con `subject recall = 1.0000` usando
> `thr=0.61`. Mejoras adicionales requieren intervencion a nivel de entrenamiento o
> arquitectura.

---

## 6. Artefactos Generados

| Archivo | Descripcion |
|---|---|
| `research/scripts/post_calibrate_layermix.py` | Script completo de post-calibracion |
| `research/research_results/CSV/metrics_v3_layermix_postcal.json` | Metricas finales |
| `research/research_results/CSV/v3_layermix_postcal_comparison.csv` | Comparacion de estrategias |
| `research/research_results/CSV/v3_layermix_hard_negative_analysis.csv` | Analisis de H26-H29 |
| `research/research_results/attention_viz/final_comparison/v3_layermix_postcal_reliability.png` | Calibration plot |
| `research/research_results/attention_viz/final_comparison/v3_layermix_postcal_thresholds.png` | Comparacion barras |
| `research/research_results/attention_viz/final_comparison/v3_layermix_postcal_hard_negatives.png` | Hard negatives |
| `FASE3F_POSTCALIBRATION_LAYERMIX.md` | Este documento |
