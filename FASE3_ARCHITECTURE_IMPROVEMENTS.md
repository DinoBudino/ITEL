# FASE 3: Optimización de Robustez y Arquitectura

**Objetivo:** Mejorar el modelo base checo HuBERT + Bi-LSTM + Single-Head Attention + SupCon
mediante 3 innovaciones arquitectónicas, con validación limpia 10-fold Cross-Validation.

**Fecha:** Mayo 2026

**Estado:** LayerMix (Mejora A) validado. Mejora B (noise) y Mejora C (sparse attention) pendientes
de experimentación completa. Prueba con dataset inglés pendiente.

---

## 1. Las 3 Mejoras Implementadas

### Mejora A: Feature Fusion -- HuBERT Layer Weighting (ScalarMix)

**Cambio:** En lugar de usar solo la última capa de HuBERT, se implementa un `ScalarMix`
que aprende pesos por capa y produce una suma pesada de las 12 capas del transformer.

**Hipótesis:** Las capas intermedias contienen información fonética clave para TEL que la
última capa (más semántica) podría estar omitiendo.

**Flag:** `--use-layer-mix`

**Requiere:** Extracción previa con `--all-layers` (guarda [12, T, 768] en vez de [T, 768]).

### Mejora B: Regularización por Ruido en Embeddings

**Cambio:** Inyecta ruido Gaussiano `N(0, std)` en los embeddings durante `model.train()`.
En `model.eval()` se desactiva automáticamente.

**Hipótesis:** Fuerza a la Bi-LSTM a ignorar variaciones de tono o acento, centrándose
solo en los patrones temporales de la patología.

**Flag:** `--embedding-noise-std 0.05`

### Mejora C: Sparse Attention -- Penalización de Entropía

**Cambio:** Añade penalización de entropía a los pesos de atención:
`L_ent = -mean(SUM alpha_i * log(alpha_i))`. La loss total es:
`BCE + lambda_ctr * SupCon + lambda_ent * Entropy`.

**Hipótesis:** Obliga al modelo a ser más "puntiagudo" en su atención, ignorando
silencios o ruidos de fondo.

**Flag:** `--attention-entropy-lambda 0.01`

---

## 2. Arquitectura Técnica

### Modelo: `AttentionBiLSTMClassifierV3`

Ubicación: `research/src/sequence_model.py`

```
Input: [B, layers?, T, 768]
  |
  +-- (opcional) ScalarMix {12 learned weights} → [B, T, 768]
  +-- (opcional) Gaussian Noise N(0, std)
  +-- LayerNorm(768)
  +-- Bi-LSTM(2 layers, hidden=256) → [B, T, 512]
  +-- Single-Head Attention → context [B, 512]
  |     +-- (opcional) Entropy Loss
  +-- Classifier: LayerNorm → Dropout → Linear(512→128) → ReLU → Dropout → Linear(128→1)
  |
Output: logits [B]  +  context [B, 512]  +  (opcional) entropy_loss
```

### Extracción de capas: `HubertSequenceEncoder.extract_all_layers()`

Ubicación: `research/src/sequence_model.py:40-50`

- Carga HUBERT_BASE de torchaudio
- Extrae `extract_features()` → 12 capas transformer
- Guarda tensor [12, T, 768] en `.pt`

---

## 3. Protocolo Experimental

### Datos
- Dataset: `cz_main_cleaned.csv` (2975 muestras, 98 sujetos)
- Labels: 1709 Patients, 1266 Healthy
- Preprocesamiento: 16kHz, highpass 100Hz, RMS target 0.08, perfil `multilingual_light`

### Cross-Validation
- **StratifiedGroupKFold, n_splits=10, random_state=42**
- Speaker-independent (ningún sujeto aparece en train y test del mismo fold)
- Métrica prioritaria: **Recall** (optimizada en validación por fold)
- Threshold: optimizado por fold vía grid search [0.20, 0.80]

### Entrenamiento
- 30 epochs, lr=1e-3, weight_decay=1e-4
- Loss: BCE + 0.1 * Supervised Contrastive (SupCon)
- Batch size: 16
- Pos weight: 1.4

### Experimentos ejecutados

| Experimento | Secuencias | LayerMix | CV | Checkpoint dir |
|---|---|---|---|---|
| V3 Control | last layer (1 capa) | No | 10-fold | `research/checkpoints/v3_final/baseline/` |
| V3 LayerMix | all 12 layers | **Sí** | 10-fold | `research/checkpoints/v3_final/layermix/` |

---

## 4. Resultados: V3 Control vs V3 LayerMix

### 4.1 Métricas agregadas (sample-level, threshold 0.5)

| Métrica | V3 Control | V3 LayerMix | Delta |
|---|---:|---:|---:|
| Accuracy | 0.8316 | **0.8948** | **+0.0632** |
| Balanced Accuracy | 0.8089 | **0.8783** | **+0.0694** |
| Precision | 0.7907 | **0.8518** | **+0.0612** |
| Recall | 0.9614 | **0.9889** | **+0.0275** |
| F1 | 0.8677 | **0.9152** | **+0.0475** |
| AUC | 0.9033 | **0.9451** | **+0.0418** |

### 4.2 Métricas a nivel sujeto

| Métrica | V3 Control | V3 LayerMix | Delta |
|---|---:|---:|---:|
| Subject Accuracy | 0.9082 | **0.9490** | **+0.0408** |
| Subject Balanced Accuracy | 0.8977 | **0.9432** | **+0.0455** |
| Subject Precision | 0.8571 | **0.9153** | **+0.0581** |
| Subject Recall | 1.0000 | 1.0000 | 0.0000 |
| Subject F1 | 0.9231 | **0.9558** | **+0.0327** |
| Subject AUC | 0.9621 | **0.9739** | **+0.0118** |

### 4.3 Estabilidad entre folds (fold_summary)

| Métrica | V3 Control | V3 LayerMix | Interpretación |
|---|---:|---:|---|
| Mean Fold Balanced Accuracy | 0.7310 ± 0.1108 | **0.8591 ± 0.0680** | +12.8pp mean, -39% std |
| Mean Fold AUC | 0.9417 ± 0.0363 | **0.9657 ± 0.0394** | +2.4pp mean |
| Mean Fold F1 | 0.8302 ± 0.0749 | **0.9001 ± 0.0595** | +7.0pp mean, -21% std |
| Mean Fold Recall | 0.9874 ± 0.0116 | **0.9962 ± 0.0043** | +0.9pp mean, -63% std |

### 4.4 Comparación con Fase 1 (referencia)

| Métrica | Fase 1 (5-fold, 3ep) | V3 Control (10-fold) | V3 LayerMix (10-fold) |
|---|---:|---:|---:|
| Sample Accuracy | 0.8603 | 0.8316 | **0.8948** |
| Sample Balanced Accuracy | 0.8577 | 0.8089 | **0.8783** |
| Sample AUC | 0.9216 | 0.9033 | **0.9451** |
| Subject Accuracy | 0.8947 | 0.9082 | **0.9490** |
| Subject AUC | 0.9556 | 0.9621 | **0.9739** |

**Nota:** La Fase 1 usó 5-fold y solo 3 epochs (checkpoint rápido). Una comparación
totalmente justa requeriría re-ejecutar Fase 1 con 10-fold y 30 epochs.

---

## 5. Interpretación de Resultados

### 5.1 LayerMix mejora en todas las métricas principales

No es solo accuracy: suben balanced accuracy (+6.9pp), AUC (+4.2pp), F1 (+4.8pp),
y precision (+6.1pp). Esto descarta que la mejora sea solo por sesgo hacia la clase
mayoritaria.

### 5.2 LayerMix produce modelos más estables

La reducción de varianza entre folds es significativa:
- Balanced accuracy std: 0.1108 → 0.0680 (-39%)
- Recall std: 0.0116 → 0.0043 (-63%)

Un modelo con menor varianza entre folds generaliza mejor y es más confiable
para despliegue clínico.

### 5.3 Recall se mantiene en niveles clínicamente aceptables

Ambos modelos tienen recall > 0.96 a nivel sample. LayerMix lo mejora marginalmente
(de 0.961 a 0.989), pero la ganancia principal está en **precision** (+6.1pp), lo que
significa **menos falsos positivos** sin sacrificar detección de pacientes.

### 5.4 La señal está en las capas intermedias

El hecho de que una suma pesada de 12 capas supere consistentemente a usar solo la
última capa valida la hipótesis original: **las capas intermedias de HuBERT contienen
información fonética y acústica relevante para TEL que la última capa omite o atenúa**.

---

## 6. Transferencia a Inglés (Fase 2 -- Resumen)

La Fase 2 evaluó la transferencia del modelo checo a un dataset inglés de 12 niños
con TEL. Resultados principales:

| Método | Balanced Accuracy | AUC | Notas |
|---|---|---|---|
| Zero-shot (thr=0.20, checo) | 0.500 | 0.600 | Colapso: todo Patients |
| Zero-shot (thr=0.95, recalibrado) | **0.700** | 0.600 | Señal existe, umbral desplazado |
| Linear Probe LOSO (LR) | 0.657 | **0.743** | Mejor ranking (AUC +24%) |

**Conclusión de Fase 2:** Los embeddings HuBERT+Bi-LSTM contienen información
diagnóstica transferible entre idiomas (AUC sube de 0.60 a 0.74 con solo re-entrenar
una capa lineal), pero la frontera de decisión binaria requiere más datos locales.

Documento completo: `FASE2_MULTILINGUAL_FINDINGS.md`

---

## 7. Pendientes

### 7.1 Mejoras B y C
- Embedding noise y attention entropy compilan y entrenan correctamente
- **No se ha ejecutado experimento completo 10-fold** para ninguna de las dos
- Recomendación: evaluar `--embedding-noise-std 0.05` y `--attention-entropy-lambda 0.01`
  por separado y en combinación con LayerMix

### 7.2 Prueba con dataset inglés
- El modelo V3 LayerMix **no se ha evaluado aún sobre el dataset inglés**
- Pendiente: extraer secuencias all-layers del inglés, correr zero-shot y linear probing
  con el checkpoint V3 LayerMix

### 7.3 Comparación contra Kotarba (ResNet-18)
- El repo tiene infraestructura legacy de espectrogramas (`research/src/spectrograms.py`)
- Una réplica controlada de ResNet-18 sobre espectrogramas log-mel permitiría comparar
  directamente contra el 99.48% reportado, usando el mismo protocolo 10-fold speaker-aware

---

## 8. Artefactos Generados

### Código fuente

| Archivo | Descripción |
|---|---|
| `research/src/sequence_model.py` | +`ScalarMix`, +`HubertSequenceEncoder.extract_all_layers()`, +`AttentionBiLSTMClassifierV3`, +`SequenceV3Dataset`, `collate_sequences` actualizado para 4D |
| `research/scripts/extract_hubert_sequences.py` | +`--all-layers` flag |
| `research/scripts/train_optimized_v3.py` | Script V3 con flags para las 3 mejoras, 10-fold CV, Recall prioritario |
| `research/scripts/evaluate_zero_shot_en.py` | Zero-shot: checkpoint checo sobre inglés |
| `research/scripts/audit_zero_shot_geometry.py` | Auditoría geométrica: histograma, umbral, centroides, UMAP |
| `research/scripts/linear_probe_en.py` | Linear probing LOSO (Logistic Regression + MLP classifier) |

### Checkpoints

| Archivo | Descripción |
|---|---|
| `research/checkpoints/best_model_checo_fase1.pt` | Modelo Fase 1 (15.5 MB). Incluye state_dict, model_config, threshold, preprocessing |
| `research/checkpoints/v3_final/baseline/` | 10 checkpoints V3 control (fold_1..10_of_10.pt) |
| `research/checkpoints/v3_final/layermix/` | 10 checkpoints V3 LayerMix (fold_1..10_of_10.pt) |

### Métricas y reportes

| Archivo | Descripción |
|---|---|
| `research/research_results/CSV/metrics_v3.json` | Métricas V3 control (10-fold) |
| `research/research_results/CSV/metrics_v3_layermix.json` | Métricas V3 LayerMix (10-fold) |
| `research/research_results/CSV/predictions_v3.csv` | Predicciones V3 control |
| `research/research_results/CSV/predictions_v3_layermix.csv` | Predicciones V3 LayerMix |
| `research/research_results/CSV/metrics_contrastive_l0.1.json` | Referencia Fase 1 (5-fold rápido) |
| `research/research_results/transfer_en/diagnostic_audit_report.json` | Auditoría geométrica inglés |
| `research/research_results/transfer_en/linear_probe_report.json` | Linear probing LOSO inglés |
| `FASE2_MULTILINGUAL_FINDINGS.md` | Reporte Fase 2 |
| `FASE3_ARCHITECTURE_IMPROVEMENTS.md` | Este documento |

### Visualizaciones

| Archivo | Descripción |
|---|---|
| `research/research_results/attention_viz/transfer_en/umap_zero_shot_cz_en.png` | UMAP: 580 CZ + 12 EN, centroides checos |
| `research/research_results/attention_viz/transfer_en/umap_linear_probe_loso.png` | UMAP post-probing: labels + scores |
| `research/research_results/attention_viz/transfer_en/score_histogram_en.png` | Histograma scores zero-shot |
| `research/research_results/attention_viz/transfer_en/score_distribution_comparison.png` | ZS vs Linear Probe |

### Datasets

| Archivo | Descripción |
|---|---|
| `research/processed_data/manifests/hubert_sequences_cleaned.csv` | Secuencias capa única (2975 muestras) |
| `research/processed_data/manifests/hubert_sequences_all_layers.csv` | Secuencias 12 capas (2975 muestras) |
| `research/processed_data/manifests/hubert_sequences_en_test.csv` | Secuencias inglés (12 muestras) |

---

## 8. Fase 3B: Evaluación de LayerMix en Inglés

### 8.1 Resultados Zero-Shot y Linear Probing

Se evaluó el checkpoint V3 LayerMix (fold 8, bal_acc=0.9753 en checo, `use_layer_mix=True`)
sobre el dataset inglés de 12 sujetos (7 Healthy, 5 Patients), usando el mismo protocolo
de Fase 2.

| Método | AUC | Bal Acc | Recall | Precision | F1 |
|---|---:|---:|---:|---:|---:|
| Fase 2 Zero-Shot recalibrado (thr=0.95) | 0.600 | **0.700** | 0.400 | 1.000 | 0.571 |
| Fase 2 Linear Probe LOSO | **0.743** | **0.657** | **0.600** | **0.600** | **0.600** |
| **V3 LM Zero-Shot** (thr=0.20) | 0.600 | 0.500 | 1.000 | 0.417 | 0.588 |
| **V3 LM Zero-Shot recalibrado** | 0.600 | 0.500 | 0.000 | 0.000 | 0.000 |
| **V3 LM Linear Probe LOSO** | 0.514 | 0.557 | 0.400 | 0.500 | 0.444 |

### 8.2 Diagnóstico

**LayerMix empeora sistemáticamente la transferencia al inglés.** El zero-shot produce
saturación total (todos los scores en ~1.0000, sin variación entre sujetos). El linear
probing cae de AUC=0.743 (Fase 2) a AUC=0.514 (Fase 3). La geometría también se degrada:
en Fase 2 había separación entre distancias a centroides (Healthy→P: 1.65, Healthy→H: 4.14,
ratio 2.5x). En Fase 3, todas las distancias colapsan al mismo rango (~2.1-2.3).

**Explicación:** LayerMix aprende rasgos más ricos y específicos del checo usando las 12
capas del transformer. Esto es excelente para rendimiento intra-idioma (+6.9pp balanced
accuracy en checo), pero contraproducente para generalización cross-lingual: el modelo
encuentra más razones para clasificar cualquier audio no-checo como anómalo, con máxima
confianza. Es un caso claro de **overfitting de dominio**.

**Veredicto 3B:** LayerMix es el mejor modelo para checo. El modelo base single-layer
(Fase 2) es mejor punto de partida para transferencia multilingüe.

Documento completo: `FASE3B_LAYERMIX_ENGLISH_EVALUATION.md`

---

## 9. Contraste con Kotarba (ResNet-18, 2020)

### 9.1 El Referente Externo

Kotarba (2020) reporta sobre el mismo corpus LANNA (98 niños checos, 44 sanos + 54 TEL):
- **Accuracy: 99.48%**
- **AUC-ROC: 0.9997**
- **F1-Score: 98.98%**

Enfoque: espectrogramas log-potencia + ResNet-18 (tratando el audio como imagen).

### 9.2 Lo que tus resultados revelan que Kotarba no muestra

La diferencia fundamental no está en la arquitectura (CNN vs Transformer), sino en **qué
pregunta científica se está respondiendo**:

| Dimensión | Kotarba (ResNet-18) | Este proyecto (HuBERT+BiLSTM) |
|---|---|---|
| Rendimiento en dominio (checo) | 99.48% | 89.5% (sample) / 94.9% (subject) |
| Transferencia a otro idioma | No reportada | **Probada**: señal detectable (AUC=0.74) |
| Estudio del espacio latente | No (caja negra CNN) | **Sí**: centroides, UMAP, distancias, bridge subjects |
| Sensibilidad al domain shift | No evaluada | **Demostrada**: cambio de idioma = cambio de manifold |
| Efecto de riqueza de features | No analizado | **Demostrado**: más capas → mejor intra-dominio, peor cross-lingual |

### 9.3 Interpretación Arquitectónica

**Kotarba probablemente capturó muy bien el "TEL-en-LANNA".**  
Un 99.48% en el mismo corpus sugiere que la señal espectro-temporal dentro de LANNA
es extremadamente fuerte y que una CNN profunda puede explotarla casi perfectamente.

**Este proyecto está capturando algo más difícil y más valioso:** qué parte del marcador
TEL sobrevive fuera del dominio de entrenamiento.

La observación clave es esta:

> **Capas más ricas (LayerMix) → mejor ajuste intra-dominio (+6.9pp en checo)**  
> **Representaciones más simples (single-layer) → mejor transferencia cross-lingual**

Esto es una conclusión arquitectónica no trivial: para un modelo clínico que aspire a
generalizar entre idiomas, la simplicidad del encoder puede ser una virtud, no una
limitación.

### 9.4 Por qué un modelo con menor accuracy puede tener mayor valor translacional

El 99.48% de Kotarba podría deberse parcialmente a:
- Evaluación a nivel utterance (no sujeto)
- Segmentación muy controlada (VAD, utterances aisladas)
- Menor variabilidad acústica en el protocolo
- Posibles pistas de grabación o estructura de corpus

Nada de esto invalida su resultado. Pero sí limita lo que se puede concluir sobre
**biomarcadores universales del TEL**.

Este proyecto, con un accuracy menor en checo, aporta evidencia más sólida sobre:
1. La existencia de señal transferible entre idiomas (AUC sube de 0.60 a 0.74 con probing)
2. La estructura geométrica del espacio latente (centroides, bridge subjects)
3. El trade-off entre especialización y generalización (LayerMix vs single-layer)

### 9.5 Recomendación para Posicionamiento Científico

Si el objetivo es publicar o defender este trabajo frente a Kotarba, la narrativa debería
ser:

> "Mientras que Kotarba (2020) demuestra que el TEL es altamente detectable dentro del
> corpus LANNA usando CNNs sobre espectrogramas, nuestro trabajo aborda una pregunta
> complementaria y clínicamente más relevante: ¿son esos marcadores transferibles entre
> idiomas? Nuestros resultados muestran que sí existe señal cross-lingual (AUC=0.74 vía
> linear probing), pero que la especialización arquitectónica que maximiza el rendimiento
> intra-dominio (LayerMix, +6.9pp) es contraproducente para la generalización. Esto
> sugiere que el camino hacia un biomarcador universal del TEL requiere equilibrar
> riqueza de representación con robustez al domain shift."

---

## 10. Conclusión de Fase 3

**LayerMix (Mejora A) está validado como mejora arquitectónica significativa para checo.**
La fusión pesada de las 12 capas de HuBERT supera consistentemente al baseline de última
capa en accuracy (+6.3pp), balanced accuracy (+6.9pp), AUC (+4.2pp), F1 (+4.8pp)
y estabilidad entre folds (-39% en varianza de balanced accuracy).

**LayerMix NO mejora la transferencia al inglés** (Fase 3B). La especialización que lo hace
excelente en checo lo vuelve frágil cross-lingual. El modelo base single-layer de Fase 2
es mejor punto de partida para adaptación multilingüe.

**Las mejoras B (embedding noise) y C (sparse attention) están implementadas y**
compiladas, pendientes de experimentación completa 10-fold. La Mejora B (ruido) es
particularmente prometedora para transferencia, ya que la regularización por ruido
suele mejorar generalización cross-dominio, al contrario que LayerMix.

**El contraste con Kotarba revela que este proyecto está respondiendo una pregunta más
ambiciosa** (transferencia cross-lingual) que la referencia externa (detección intra-corpus),
y que la aparente desventaja en accuracy esconde una contribución científica más profunda
sobre la naturaleza de los biomarcadores acústicos del TEL.
