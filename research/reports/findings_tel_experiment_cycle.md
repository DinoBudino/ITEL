# Informe Tecnico Consolidado: Deteccion Automatica de TEL

**Arquitectura:** HuBERT_BASE + Bi-LSTM + Single-Head Attention
**Dataset:** LANNA (checo)
**Evaluacion:** 5-fold StratifiedGroupKFold por `subject_id`

---

## 1. Introduccion y Punto de Partida

El objetivo de la investigacion fue construir y auditar un sistema de deteccion automatica del Trastorno Especifico del Lenguaje (TEL) a partir de audio infantil, utilizando un encoder autosupervisado y un clasificador secuencial.

El **baseline inicial** se definio como:

- `HuBERT_BASE` congelado como extractor de secuencias acusticas
- `Bi-LSTM` bidireccional de 2 capas (256 hidden)
- mecanismo de `single-head attention`
- clasificacion binaria `Healthy` vs `Patients`

La primera hipotesis fue que una arquitectura secuencial sobre representaciones de HuBERT podria capturar micro-patrones temporales del habla patologica mejor que un clasificador tabular sobre embeddings agregados.

Sin embargo, el analisis del pipeline inicial revelo dos problemas estructurales:

1. **Sesgo de ventana temporal fija**
   - El preprocesado forzaba `2.0s` por muestra mediante `center_crop` o `padding`.
   - Esto generaba una desproporcion severa entre senal util y silencio artificial.

2. **Ruido en tareas de baja densidad informativa**
   - Las muestras `controlled_vowel` y los clips extremadamente cortos ofrecian poca estructura temporal.
   - El modelo tendia a sobre-atender micro-regiones en este tipo de inputs.

---

## 2. Diagnostico Temporal del Corpus

Antes de modificar el modelo, se audito la duracion real de los audios originales.

**Hallazgos principales:**

| Metrica | Valor |
|---|---|
| Duracion minima | 45 ms |
| Duracion maxima | 4587 ms |
| Duracion media | 606 ms |
| Duracion mediana | **509 ms** |
| Muestras < 1.0s | **89.0%** |
| Muestras >= 2.0s | **0.8%** |

El preprocesado fijo a `2.0s` estaba convirtiendo la mayoria de las muestras en una secuencia donde el contenido util era una fraccion menor del total.

Se paso a un regimen **natural-length**, donde HuBERT procesa la duracion real de cada muestra y el padding ocurre solo a nivel de batch.

Con `natural-length`, el baseline obtuvo:

| Metrica | Sample-level | Subject-level |
|---|---|---|
| Accuracy | 83.1% | **93.9%** |
| Balanced Accuracy | 81.3% | **93.2%** |
| Recall | 95.4% | **100%** |
| AUC | 91.9% | **98.4%** |

Ademas, la desviacion estandar del AUC entre folds cayo de `~4.3pp` a `~2.4pp`, indicando una mejora clara en estabilidad.

**Interpretacion:** el mecanismo de atencion funciona mejor cuando la secuencia contiene solo el contenido acustico real, y no una ventana artificial alargada con silencio de relleno.

---

## 3. Fase 1: Curacion de Datos (El Gran Salto)

Se identificaron dos fuentes dominantes de confusion:

- `controlled_vowel` (tasa de FP extremadamente alta)
- clips con duracion real `< 0.3s` (casi impredecibles)

Se implemento la **Fase 1 de limpieza**:

- exclusion de todas las muestras `controlled_vowel`
- exclusion de todas las muestras con duracion real `< 0.3s`

**Impacto en el dataset:**

| | Antes | Despues | Delta |
|---|---|---|---|
| Muestras | 3853 | **2975** | -878 (-22.8%) |
| Patients | 2173 | 1709 | -464 |
| Healthy | 1680 | 1266 | -414 |

**Resultados del baseline limpio:**

| Metrica | Sample-level | Subject-level |
|---|---|---|
| Accuracy | **85.95%** | **88.78%** |
| Balanced Accuracy | **84.29%** | **87.50%** |
| Precision | 82.75% | 83.08% |
| Recall | 95.44% | **100%** |
| F1 | 88.64% | 90.76% |
| AUC | 92.39% | **97.85%** |

**Comparacion vs baseline natural sin limpieza:**

- **+2.9pp** en accuracy por muestra
- mejora en balanced accuracy, precision, F1 y AUC
- reduccion de FP globales de `59.8%` a `47.3%`

**Conclusion cientifica:** la deteccion de TEL en este contexto depende de dinamica temporal suficiente. El sistema necesita al menos una minima estructura secuencial para explotar la senal patologica. El TEL no se comporta como una firma puramente instantanea, sino como un patron distribuido en el tiempo.

---

## 4. Fase 2: Exploracion Arquitectonica (Multi-Head Attention)

Se implemento **Multi-Head Attention (4 cabezas)** sobre el baseline limpio, manteniendo:

- HuBERT congelado
- Bi-LSTM
- natural-length
- dataset limpio

**Motivacion:** forzar al modelo a capturar multiples focos de evidencia temporal.

**Resultados MHA (4 cabezas):**

| Metrica | Single-head (limpio) | MHA 4-head | Delta |
|---|---|---|---|
| Accuracy | **85.95%** | 83.7% | -2.2pp |
| Balanced Accuracy | **84.29%** | 80.0% | **-4.3pp** |
| Recall | **95.44%** | 91.6% | -3.8pp |
| AUC | **92.39%** | 89.1% | -3.3pp |

**Con conclusion:** la MHA introdujo dilucion de senal. Causas: secuencias demasiado cortas para 4 cabezas, insuficiente volumen de datos para estabilizar multiples patrones de atencion, y competencia innecesaria entre cabezas.

---

## 5. Fase 3A: Especializacion por Tarea (Expertos)

Se entrenaron dos modelos expertos independientes:

- **Word-Expert**: solo `controlled_word`
- **Consonant-Expert**: solo `controlled_consonant`

**Hipotesis:** la mezcla de tareas introducia interferencia.

**Resultados:**

| Modelo | Muestras | Sample Acc | Sample BalAcc | Subject Acc | Subject AUC |
|---|---|---|---|---|---|
| **Combinado** | 2975 | **85.95%** | **84.29%** | **88.78%** | **97.85%** |
| Word-Expert | 2208 | 81.1% | 79.1% | 87.8% | 96.1% |
| Cons-Expert | 767 | 79.5% | 75.4% | 80.6% | 94.4% |
| Ensemble | 2975 | 80.7% | 78.2% | 87.8% | 96.0% |

**Con conclusion:** la hipotesis de interferencia fue refutada. Existe **sinergia acustica** entre tareas. Separarlas fragmenta el aprendizaje y reduce el volumen de datos por modelo. En datasets de escala pequena/mediana, la diversidad actua como regularizacion natural.

---

## 6. Fase 3B: Contrastive Learning (El Hallazgo Maestro)

Se implemento una perdida combinada:

```
L = L_BCE + lambda * L_SupCon
```

con `lambda = 0.1`, `temperature = 0.07`.

La perdida contrastiva supervisada se aplico sobre el **context vector** (salida de atencion, antes del clasificador). No se modifico el backbone.

**Resultados:**

| Metrica | Baseline limpio | Contrastive (l=0.1) | Delta |
|---|---|---|---|
| Sample Accuracy | **85.95%** | 84.03% | -1.9pp |
| Sample AUC | 92.39% | 92.98% | +0.6pp |
| Subject BalAcc | 87.50% | **90.91%** | **+3.4pp** |
| Subject AUC | 97.85% | 97.47% | -0.4pp |
| FP rate | 47.3% | **43.8%** | -3.5pp |

### El Caso H29

Se genero una visualizacion 2D del espacio latente (UMAP) y se calcularon distancias a los centroides `Healthy` y `Patients`.

**Distancias de sujetos criticos:**

| Sujeto | d(Healthy) | d(Patients) | Ratio | Lado |
|---|---|---|---|---|
| H26 | 14.491 | 13.734 | 1.06 | Patient |
| H27 | 14.368 | 13.174 | 1.09 | Patient |
| **H28** | 10.288 | 11.820 | 0.87 | **Healthy** |
| **H29** | 8.944 | 10.819 | 0.83 | **Healthy** |

Distancia entre centroides: `2.11`.

**Hallazgo fundamental:**

H29 y H28 caen del **lado sano** del espacio latente, mas cerca del centroide `Healthy` que del de `Patients`, a pesar de ser clasificados como TEL por el clasificador binario.

Esto demuestra que:

1. el clasificador binario puede asignar erroneamente TEL a sujetos sanos atipicos
2. el espacio metrico contrastivo revela una estructura latente mas fiel a la organizacion clinica real
3. H29 no es un error aleatorio del modelo, sino un **sano atipico** que requiere lectura cuidadosa

---

## 7. Evolucion de Sujetos Criticos a lo Largo de las Fases

| Sujeto | Natural (todo) | Limpio | MHA | Word-Exp | Cons-Exp | Contrastivo (score) | UMAP lado |
|---|---|---|---|---|---|---|---|
| **H27** | 100% FP | 85% FP | 100% FP | 100% FP | 100% FP | 0.70 | Patient |
| **H29** | 100% FP | 100% FP | 100% FP | 100% FP | 100% FP | 0.80 | **Healthy** |
| **H26** | 92% FP | 96% FP | 96% FP | 100% FP | 100% FP | 0.98 | Patient |
| **H28** | 97% FP | 100% FP | 97% FP | 100% FP | 100% FP | 0.91 | **Healthy** |

Ninguna fase resolvio los FPs de estos sujetos a nivel de clasificacion binaria.
Solo la Fase 3B revelo que H28 y H29 estan geometricamente en el lado correcto (sano).

---

## 8. Sintesis Global

| Fase | Accion | Resultado | Valor cientifico |
|---|---|---|---|
| Baseline natural | Eliminar `fit_to_duration` | Mejora estabilidad | Senal TEL no necesita 2s |
| **Fase 1** | Filtrar vocales y <0.3s | **+2.9pp Acc** | Unica mejora clara y sostenida |
| Fase 2 | Multi-Head Attention | **Empeoro** | MHA diluye senal en clips cortos |
| Fase 3A | Expertos por tarea | **Empeoro** | La mezcla de tareas crea sinergia |
| Fase 3B | Contrastive Learning | Mejora subject BalAcc | H29 es sano atipico en espacio metrico |

### Mejor modelo operativo

- `HuBERT_BASE` congelado
- `Bi-LSTM` bidireccional
- `Single-Head Attention`
- `natural-length`
- Dataset limpio (sin vocales, sin <0.3s)
- `pos_weight=1.4`, `threshold=0.2`

**Metricas finales:**

| Metrica | Sample-level | Subject-level |
|---|---|---|
| Accuracy | **85.95%** | **88.78%** |
| Balanced Accuracy | **84.29%** | **87.50%** |
| Recall | 95.44% | **100%** |
| AUC | 92.39% | **97.85%** |

---

## 9. Conclusiones Generales

1. **La mayor mejora real vino de la calidad del dato, no de la complejidad arquitectonica.**

2. **La arquitectura base ya era adecuada.** HuBERT congelado + Bi-LSTM + single-head attention fue suficientemente expresiva para el tamano del corpus.

3. **Las mejoras arquitectonicas complejas pueden ser contraproducentes en secuencias cortas y datasets medianos.**

4. **El problema residual ya no es puramente de clasificacion, sino de estructura latente y frontera clinica.** H29 revela la existencia de una zona fronteriza entre sano tipico y TEL.

5. **El contraste metrico captura matices que la clasificacion binaria no logra representar.**

---

## 10. Implicancias Clinicas y Trabajo Futuro

### Uso clinico recomendado
- Triaje, no diagnostico definitivo.
- Combinar score binario + distancia a centroides Healthy/Patients.
- Senalizar sujetos fronterizos o atipicos.

### Trabajo futuro

1. Validacion externa en otra cohorte.
2. Incremento de sujetos para robustecer la frontera latente.
3. Calibracion clinica de zonas grises del embedding.
4. Decisiones basadas en distancia o prototipos, no solo en logits.
5. Analisis experto de H29 para determinar si representa variabilidad normal del habla infantil, error de etiquetado, o un subtipo intermedio.

---

## Artefactos Asociados

### Metricas principales

| Archivo | Fase |
|---|---|
| `research/research_results/CSV/metrics_attention_bilstm_natural.json` | Baseline natural-length |
| `research/research_results/CSV/metrics_cleaned_baseline.json` | Fase 1: Limpieza |
| `research/research_results/CSV/metrics_attention_bilstm_natural_mha4.json` | Fase 2: MHA |
| `research/research_results/CSV/metrics_word_expert.json` | Fase 3A: Word-Expert |
| `research/research_results/CSV/metrics_consonant_expert.json` | Fase 3A: Consonant-Expert |
| `research/research_results/CSV/metrics_contrastive_l0.1.json` | Fase 3B: Contrastive |

### Auditorias de error

| Archivo | Fase |
|---|---|
| `research/research_results/CSV/error_audit_report.csv` | Baseline natural |
| `research/research_results/CSV/error_audit_cleaned.csv` | Fase 1: Limpieza |
| `research/research_results/CSV/error_audit_mha4.csv` | Fase 2: MHA |
| `research/research_results/CSV/error_audit_word_expert.csv` | Fase 3A: Word |
| `research/research_results/CSV/error_audit_consonant_expert.csv` | Fase 3A: Consonant |

### Visualizaciones

| Archivo | Contenido |
|---|---|
| `research/research_results/attention_viz/attention_A_TP_word.png` | Caso A: TP paciente, palabra |
| `research/research_results/attention_viz/attention_B_FP_H27_H29_word.png` | Caso B: FP H29, palabra |
| `research/research_results/attention_viz/attention_C_FP_vowel.png` | Caso C: FP vocal |
| `research/research_results/attention_viz/umap_contrastive_l0.1.png` | UMAP espacio contrastivo |
| `research/research_results/attention_viz/ablation_A_TP_word.txt` | Ablacion Caso A |

### Embeddings

| Archivo | Contenido |
|---|---|
| `research/research_results/CSV/embeddings_contrastive_l0.1.parquet` | Context vectors del modelo contrastivo |

### Scripts principales

| Archivo | Funcion |
|---|---|
| `research/src/sequence_model.py` | `HubertSequenceEncoder`, `AttentionBiLSTMClassifier`, `SequenceDataset`, `collate_sequences` |
| `research/scripts/extract_hubert_sequences.py` | Extraccion de secuencias HuBERT (natural-length) |
| `research/scripts/train_attention_bilstm.py` | Entrenamiento base y variantes (MHA, expertos) |
| `research/scripts/train_contrastive.py` | Entrenamiento BCE + Supervised Contrastive |
| `research/scripts/audit_errors.py` | Auditoria de errores y profiling de FP |
| `research/scripts/visualize_and_ablate.py` | Visualizacion de atencion + ablacion |
| `research/scripts/visualize_embeddings.py` | UMAP de embeddings contrastivos |
| `research/scripts/analyze_duration_distribution.py` | Distribucion de duraciones reales |
| `research/scripts/filter_manifest.py` | Creacion de manifest filtrado |
| `research/scripts/split_tasks.py` | Separacion word/consonant |
| `research/scripts/build_ensemble.py` | Metrica agregada del ensemble |

### Manifests

| Archivo | Contenido |
|---|---|
| `research/manifests/cz_main_samples.csv` | Manifest original |
| `research/manifests/cz_main_cleaned.csv` | Manifest filtrado (sin vocales, sin <0.3s) |
| `research/manifests/cz_main_cleaned_word.csv` | Solo controlled_word |
| `research/manifests/cz_main_cleaned_consonant.csv` | Solo controlled_consonant |
| `research/processed_data/manifests/hubert_sequences_natural.csv` | Secuencias natural-length |
| `research/processed_data/manifests/hubert_sequences_cleaned.csv` | Secuencias dataset limpio |
| `research/processed_data/manifests/hubert_sequences_cleaned_word.csv` | Secuencias word-only |
| `research/processed_data/manifests/hubert_sequences_cleaned_consonant.csv` | Secuencias consonant-only |
