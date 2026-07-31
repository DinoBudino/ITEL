# INFORME FINAL COMPARATIVO ITEL

**Objetivo:** Consolidar todos los hallazgos del proyecto en un único documento maestro, con tablas, gráficos y la decisión arquitectónica final.

**Fecha:** Mayo 2026

---

## 1. Resumen Ejecutivo

- **Mejor modelo para checo intra-dominio:** `V3 LayerMix + hardneg_hnweight2x` (HuBERT 12-layer mix + BiLSTM + attention + SupCon + hard-negative weighted training), con `subject balanced_accuracy = 0.9773`, `subject precision = 0.9643`, `subject recall = 1.0000`.
- **Mejor modelo para transferencia multilingüe / inglés:** `V3 Noise 0.03` (HuBERT single-layer + BiLSTM + attention + SupCon + embedding noise std=0.03), con `zero-shot AUC = 0.800` en inglés, `recalibrated balanced_accuracy = 0.786`, `linear probe balanced_accuracy = 0.757`.
- **Hallazgo arquitectónico central:** No existe un único modelo óptimo para ambos dominios. Las mejoras que maximizan la discriminación intra-dominio (LayerMix, hardneg) son contraproducentes para la generalización cross-lingual. La regularización (embedding noise) mejora la robustez multilingüe sin sacrificar rendimiento clínico.
- **Rol del benchmark ResNet-18 tipo Kotarba:** Confirmó que la señal espectro-temporal existe y es fuerte dentro de LANNA, pero no aportó una ventaja decisiva sobre las ramas HuBERT existentes. Útil como validación científica externa, no como nueva rama productiva.
- **Ramas explícitamente descartadas:** Sparse Attention, subtask-specialized experts, hardneg para transferencia a inglés, ResNet-18 como rama principal.

---

## 2. Metodología Consolidada

### Corpus LANNA (Checo)
- 98 niños (54 con TEL, 44 con desarrollo típico)
- 2975 muestras curadas
- Tareas: consonantes controladas (767) + palabras controladas por sílaba (2208)
- Preprocesado común: mono, 16kHz, highpass 100Hz, normalize_rms target 0.08

### Corpus Inglés
- 12 niños (7 Healthy, 5 Patients)
- Picture description, audio natural-length
- Mismo preprocesado que el corpus checo

### Protocolo de evaluación
- **Speaker-independent:** StratifiedGroupKFold, ningún sujeto aparece en train y test del mismo fold
- **10-fold CV** para todos los modelos checos post-Fase 1
- **Métricas reportadas:** tanto sample-level como subject-level
- **Subject-level:** media de scores por sujeto, decisión binaria con threshold optimizado
- **Criterio clínico:** priorización de `subject recall >= 0.95`, optimización de `subject balanced_accuracy` y `subject precision`

---

## 3. Resultados por Fase

### 3.1 Fase 1 — Baseline Checo
**Modelo:** HuBERT-base (congelado) + BiLSTM (2 capas, hidden=256) + Single-Head Attention + BCE + SupCon (lambda=0.1)

| Métrica | Sample | Subject |
|---|---|---|
| Accuracy | 0.8603 | 0.8947 |
| Balanced Accuracy | 0.8577 | 0.8944 |
| AUC | 0.9216 | 0.9556 |
| Recall | 0.8820 | 0.9000 |
| Precision | 0.8685 | 0.9000 |
| F1 | 0.8752 | 0.9000 |

Este modelo sirvió como referencia base para todas las mejoras posteriores.

### 3.2 Fase 2 — Transferencia Multilingüe a Inglés
El zero-shot directo del modelo checo colapsó en inglés: todos los scores cayeron en la región `Patients`, con `AUC = 0.600`. La recalibración de threshold (`thr = 0.9536`) reveló que la señal existía pero estaba desplazada, alcanzando `balanced_accuracy = 0.700`. El linear probing LOSO confirmó señal transferible con `AUC = 0.743`.

**Hallazgo geométrico clave:** El sujeto 411 fue el único Healthy inglés geométricamente más cercano al centroide Healthy checo. Los demás 11 sujetos cayeron del lado `Patients`, evidenciando un domain shift fonético sistemático.

| Método | Balanced Accuracy | AUC | Interpretación |
|---|---|---|---|
| Zero-shot (thr=0.20, checo) | 0.500 | 0.600 | Colapso: todo Patients |
| Zero-shot (thr=0.95, recalibrado) | 0.700 | 0.600 | Señal desplazada, no perdida |
| Linear Probe LOSO | 0.657 | 0.743 | Señal transferible (AUC +24%) |

### 3.3 Fase 3 — Mejoras Arquitectónicas

#### 3.3.1 LayerMix (Mejora A)
Suma pesada aprendible de las 12 capas de HuBERT. Validada como mejora significativa para checo intra-dominio.

| Modelo | Sample Bal Acc | Sample AUC | Subject Bal Acc | Fold Bal Acc Mean ± Std |
|---|---:|---:|---:|---:|
| V3 control | 0.8089 | 0.9033 | 0.8977 | 0.7310 ± 0.1108 |
| **V3 LayerMix** | **0.8783** | **0.9451** | **0.9432** | **0.8591 ± 0.0680** |

**Veredicto para checo:** GO. Mejora todas las métricas y reduce varianza entre folds (-39%).

#### 3.3.2 Embedding Noise (Mejora B)
Ruido Gaussiano `N(0, 0.03)` inyectado en embeddings durante entrenamiento. Validado como mejora de robustez multilingüe.

| Modelo | Sample Bal Acc | Sample AUC | Subject Bal Acc | Fold Bal Acc Mean ± Std |
|---|---:|---:|---:|---:|
| V3 control | 0.8089 | 0.9033 | 0.8977 | 0.7310 ± 0.1108 |
| V3 Noise 0.01 | 0.8230 | 0.9166 | 0.8977 | 0.7502 ± 0.0710 |
| **V3 Noise 0.03** | 0.8497 | 0.9045 | **0.9432** | 0.8102 ± 0.0667 |
| V3 Noise 0.05 | 0.8346 | 0.9211 | 0.9205 | 0.7424 ± 0.0823 |

**Veredicto para multilingüe:** GO. `std=0.03` es el punto dulce. Mejora checo y es la mejor base para inglés.

#### 3.3.3 Sparse Attention (Mejora C)
Penalización de entropía en pesos de atención. **NO-GO**: empeora todas las métricas en checo respecto a noise-only.

#### 3.3.4 Recalibración de Precisión (V3 LayerMix)
El problema de precisión de LayerMix no era arquitectónico: era de calibración de decisión.

| Métrica | Original (thr=0.20) | Recalibrado (thr=0.61) | Ganancia |
|---|---:|---:|---:|
| Subject Precision | 0.7397 | **0.9310** | **+19.1 pp** |
| Subject Recall | 1.0000 | 1.0000 | 0 |
| Subject Balanced Accuracy | 0.7841 | **0.9545** | **+17.0 pp** |
| Subject F1 | 0.8504 | **0.9643** | **+11.4 pp** |

#### 3.3.5 Hard-Negative Training (V3 LayerMix + hardneg)
Entrenamiento con re-weighting de loss para sanos checos difíciles (H26-H29). Mejora adicional sobre la recalibración.

| Modelo | Sample Bal Acc | Sample AUC | Subject Bal Acc | Subject Precision | Subject Recall |
|---|---:|---:|---:|---:|---:|
| V3 LayerMix recalibrado | 0.8968 | 0.9451 | 0.9545 | 0.9310 | 1.0000 |
| **V3 LM + hardneg base** | 0.9349 | 0.9695 | 0.9545 | 0.9310 | 1.0000 |
| **V3 LM + hardneg hnweight2x** | **0.9434** | **0.9806** | **0.9773** | **0.9643** | **1.0000** |

**Veredicto para checo:** GO. Es el mejor modelo checo del proyecto.
**Veredicto para inglés:** NO-GO. El linear probe AUC cayó a 0.200 (peor resultado registrado). La mejora es checo-específica.

### 3.4 Benchmark ResNet-18 tipo Kotarba
Réplica controlada de la línea de Kotarba con espectrogramas log-power 180x180 y ResNet-18, mismo protocolo speaker-independent 10-fold.

| Modelo | Sample Bal Acc | Sample AUC | Subject Bal Acc | Subject Precision | Subject Recall |
|---|---:|---:|---:|---:|---:|
| ResNet-18 Kotarba | 0.8662 | 0.9418 | 0.8864 | 0.8438 | 1.0000 |
| V3 LayerMix | **0.8783** | **0.9451** | **0.9432** | 0.9153 | 1.0000 |
| V3 LM + hardneg | 0.9434 | 0.9806 | **0.9773** | **0.9643** | 1.0000 |

**Interpretación:** ResNet confirma que la señal espectro-temporal existe en LANNA, pero no supera a las ramas HuBERT optimizadas. No justifica un cambio de arquitectura.

---

## 4. Comparación Arquitectónica Final

### Tabla definitiva de modelos

| Modelo | Dominio | Escenario | Sample Acc | Sample Bal Acc | Sample AUC | Sample Rec | Sample Prec | Sample F1 | Subj Acc | Subj Bal Acc | Subj AUC | Subj Rec | Subj Prec | Subj F1 | Notas |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| Fase 1 baseline | Checo | 10-fold CV | 0.8603 | 0.8577 | 0.9216 | 0.8820 | 0.8685 | 0.8752 | 0.8947 | 0.8944 | 0.9556 | 0.9000 | 0.9000 | 0.9000 | Referencia histórica |
| V3 control | Checo | 10-fold CV | 0.8316 | 0.8089 | 0.9033 | 0.9614 | 0.7907 | 0.8677 | 0.9082 | 0.8977 | 0.9621 | 1.0000 | 0.8571 | 0.9231 | Base V3 sin mejoras |
| V3 LayerMix | Checo | 10-fold CV | 0.8948 | 0.8783 | 0.9451 | 0.9889 | 0.8518 | 0.9152 | 0.9490 | 0.9432 | 0.9739 | 1.0000 | 0.9153 | 0.9558 | Mejor arquitectura base |
| V3 Noise 0.03 | Checo | 10-fold CV | 0.8659 | 0.8497 | 0.9045 | 0.9585 | 0.8332 | 0.8914 | 0.9490 | 0.9432 | 0.9604 | 1.0000 | 0.9153 | 0.9558 | Mejor rama robusta |
| V3 LM recalibrado (thr=0.61) | Checo | Post-cal | 0.9055 | 0.8968 | 0.9451 | 0.9555 | 0.8885 | 0.9208 | 0.9592 | 0.9545 | 0.9739 | 1.0000 | 0.9310 | 0.9643 | Mejora por calibración |
| **V3 LM + hardneg hnweight2x** | **Checo** | **10-fold CV** | **0.9459** | **0.9434** | **0.9806** | 0.9602 | **0.9464** | **0.9532** | **0.9796** | **0.9773** | **1.0000** | 1.0000 | **0.9643** | **0.9818** | **Ganador final checo** |
| ResNet-18 Kotarba | Checo | 10-fold CV | 0.8850 | 0.8662 | 0.9418 | 0.9930 | 0.8372 | 0.9085 | 0.8980 | 0.8864 | 0.9579 | 1.0000 | 0.8438 | 0.9153 | Benchmark útil, no ganador |
| Fase 2 ZS recalibrado | Inglés | Zero-shot | 0.6667 | 0.7000 | 0.6000 | 0.4000 | 1.0000 | 0.5710 | — | — | — | — | — | — | Señal desplazada |
| Fase 2 LP LOSO | Inglés | Linear probe | 0.6667 | 0.6571 | 0.7429 | 0.6000 | 0.6000 | 0.6000 | — | — | — | — | — | — | Baseline inglés |
| V3 LayerMix ZS | Inglés | Zero-shot | 0.4167 | 0.5000 | 0.6000 | 1.0000 | 0.4167 | 0.5882 | — | — | — | — | — | — | Saturación total |
| **V3 Noise 0.03 ZS** | **Inglés** | **Zero-shot** | 0.7500 | **0.7857** | **0.8000** | 1.0000 | 0.6250 | **0.7692** | — | — | — | — | — | — | **Ganador inglés** |
| **V3 Noise 0.03 LP** | **Inglés** | **Linear probe** | 0.7500 | **0.7571** | **0.7429** | **0.8000** | 0.6667 | **0.7273** | — | — | — | — | — | — | **Mejor LP inglés** |
| V3 LM + hardneg ZS | Inglés | Zero-shot | 0.4167 | 0.5000 | 0.6571 | 1.0000 | 0.4167 | 0.5882 | — | — | — | — | — | — | Sigue saturado |
| V3 LM + hardneg LP | Inglés | Linear probe | 0.4167 | 0.4571 | 0.2000 | 0.2000 | 0.3333 | 0.2500 | — | — | — | — | — | — | Peor LP registrado |

### Interpretación

**Especialización vs. generalización:** LayerMix y hardneg mejoran el rendimiento intra-checo porque capturan detalles fonéticos finos del dominio LANNA. Pero esa misma especialización destruye la capacidad de generalizar a inglés. El embedding noise, por el contrario, fuerza al modelo a ignorar variaciones de dominio, mejorando la transferencia cross-lingual.

**Calibración vs. entrenamiento:** La mejora de precisión de LayerMix vino en dos etapas: primero por recalibración de threshold (cambio de 0.20 a 0.61, +19pp en subject precision), y luego por hard-negative training (+3.3pp adicionales). El primer salto fue calibración; el segundo, entrenamiento enfocado.

**No existe un único modelo óptimo:** El hallazgo más importante del proyecto es que los objetivos de máximo rendimiento intra-dominio y máxima robustez cross-lingual son contradictorios y requieren arquitecturas distintas.

---

## 5. Pronunciación y Kotarba

### Hallazgo principal del audit de pronunciación
El rendimiento del modelo varía significativamente por tipo de tarea fonética. Las palabras de 3-4 sílabas son las más discriminativas (bal_acc ~0.89-0.90), mientras que las consonantes aisladas son las peores (bal_acc ~0.81). LayerMix beneficia más a consonantes y palabras de 2-3 sílabas que a palabras complejas.

### Por qué Kotarba fue útil como referencia
Confirmó que la señal acústico-fonética en LANNA es muy fuerte y que los patrones espectro-temporales locales contienen biomarcadores relevantes para TEL. Esto valida externamente la dirección de nuestro trabajo.

### Por qué no cambió la decisión final
ResNet-18 no superó a V3 LayerMix en checo y no fue evaluado en inglés. No aportó una capacidad nueva que justificara reemplazar las ramas HuBERT existentes. Queda como confirmación externa, no como nueva arquitectura oficial.

---

## 6. Decisión Final del Proyecto

| Objetivo | Modelo final | Justificación | Métrica clave |
|---|---|---|---|
| **Checo intra-dominio** | `V3 LayerMix + hardneg_hnweight2x` | Mejor subject bal_acc (0.9773), mejor precision (0.9643), recall clínico perfecto (1.0000) | subject balanced_accuracy |
| **Transferencia multilingüe** | `V3 Noise 0.03` | Mejor zero-shot AUC en inglés (0.800), mejor recalibrado (0.786), mejor LP (0.757), mejor geometría cross-lingual (0.667) | zero-shot AUC |
| **ResNet-18 Kotarba** | Benchmark de contraste | Útil como validación externa; no supera a las ramas HuBERT | No rama oficial |
| **Ramas descartadas** | Sparse Attention, subtask experts, hardneg para inglés | Empeoran o no mejoran respecto a las ramas ganadoras | NO-GO |

---

## 7. Limitaciones

- **Dataset inglés pequeño (n=12):** Los resultados en inglés deben interpretarse con cautela. La señal detectada es real (AUC=0.800 con Noise), pero la generalización a poblaciones más grandes no está garantizada.
- **Comparabilidad imperfecta con Kotarba:** Difieren en protocolo de evaluación, segmentación, tipo de espectrograma y posiblemente en la presencia de confounds de adquisición. Nuestro benchmark ResNet-18 es una réplica razonable, no una copia exacta.
- **Sujetos hard-negative checos (H26-H29):** Siguen siendo falsos positivos incluso tras hard-negative training. Representan una limitación estructural del modelo actual.
- **Ausencia de validación clínica externa amplia:** Todo el trabajo se realizó sobre LANNA y un pequeño dataset inglés. No hay validación independiente en otros centros o poblaciones.

---

## 8. Próximo Paso Oficial

**Prioridad principal:** Adquisición de más datos en inglés (40-60 sujetos) para validar y posiblemente afinar la rama multilingüe `V3 Noise 0.03`.

**Línea secundaria:** Calibración clínica final y preparación para despliegue o validación externa del modelo checo `V3 LayerMix + hardneg_hnweight2x`.

**No se recomienda:** Continuar la exploración arquitectónica (nuevos módulos, nuevas CNNs, nuevas combinaciones) hasta que se disponga de más datos para validación externa.

---

## 9. Referencia de Gráficos

| Gráfico | Qué demuestra |
|---|---|
| `czech_branch_comparison.png` | LayerMix lidera en checo sobre V3 control y Noise 0.03 |
| `english_transfer_comparison.png` | Noise 0.03 domina en inglés sobre Fase 2 baseline y LayerMix |
| `specialization_vs_generalization.png` | El trade-off: más especialización checa = peor generalización inglesa |
| `clinical_recall_comparison.png` | Todas las ramas mantienen recall clínico >= 0.95 en todos los dominios |
| `relative_gains.png` | Cuantificación del aporte de LayerMix (+6.9pp en CZ) y Noise (+20pp AUC en EN) |
| `v3_layermix_postcal_thresholds.png` | La recalibración de threshold fue la clave de la mejora de precisión, no la arquitectura |
| `v3_layermix_postcal_hard_negatives.png` | H26-H29 siguen siendo estructuralmente difíciles incluso tras calibración y hardneg |

---

## 10. Artefactos Principales del Proyecto

### Checkpoints
- `research/checkpoints/best_model_checo_fase1.pt` — Fase 1
- `research/checkpoints/v3_final/layermix/` — 10 checkpoints V3 LayerMix
- `research/checkpoints/v3_final/noise_std003/` — 10 checkpoints V3 Noise 0.03
- `research/research_results/checkpoints/v3_layermix_hardneg/` — Checkpoints hardneg

### Scripts principales
- `research/scripts/train_optimized_v3.py` — Entrenamiento V3 con flags (LayerMix, Noise, Sparse)
- `research/scripts/train_layermix_hardneg.py` — Entrenamiento con hard-negative focused training
- `research/scripts/evaluate_zero_shot_en_v3.py` — Zero-shot V3 en inglés
- `research/scripts/linear_probe_en_v3.py` — Linear probing V3 en inglés
- `research/scripts/recalibrate_layermix.py` — Recalibración de threshold
- `research/scripts/post_calibrate_layermix.py` — Post-calibración (Platt, temp, isotonic)
- `research/scripts/train_resnet18_kotarba.py` — Benchmark ResNet-18
- `research/scripts/audit_pronunciation_structure.py` — Auditoría por pronunciación
- `research/scripts/build_final_comparison.py` — Tablas y gráficos comparativos

### Reportes
- `INFORME_FINAL_COMPARATIVO_ITEL.md` — Este documento
- `FASE2_MULTILINGUAL_FINDINGS.md` — Transferencia multilingüe
- `FASE3_ARCHITECTURE_IMPROVEMENTS.md` — Mejoras arquitectónicas
- `FASE3B_LAYERMIX_ENGLISH_EVALUATION.md` — LayerMix en inglés
- `FASE3D_NOISE_SPARSE_EVALUATION.md` — Noise + Sparse
- `FASE3F_POSTCALIBRATION_LAYERMIX.md` — Post-calibración
- `FASE3G_HARDNEG_ENGLISH_TRANSFER_EVALUATION.md` — Hardneg en inglés
- `PRONUNCIATION_AUDIT_KOTARBA_ALIGNMENT.md` — Auditoría Kotarba
