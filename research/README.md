# Research

Esta carpeta concentra el trabajo experimental de la tesis y documenta la linea activa de investigacion del proyecto. La referencia vigente se organiza alrededor de dos modelos finales:

- `V3 LayerMix (Czech)`: modelo secuencial principal para el dominio cerrado sobre LANNA.
- `Unified Multilingual (logreg_pca + centroid_shift)`: modelo ligero unificado para evaluar robustez bilingue Czech-English.

En este documento, "modelo final" identifica la configuracion experimental seleccionada. La evaluacion LOSO reajusta scaler, PCA, clasificador, centroides y threshold en cada fold; por tanto, sus folds no son un modelo unico desplegable. La promocion a bundles congelados es una etapa posterior documentada en `docs/architecture/model-lifecycle.md`.

## Estructura

```text
research/
├─ scripts/                  # Entrenamiento, evaluacion, auditorias y generacion de figuras
├─ src/                      # Codigo compartido del pipeline experimental
├─ manifests/                # Manifiestos de muestras CZ y EN
├─ processed_data/           # Secuencias HuBERT y artefactos intermedios
├─ research_results/
│  ├─ CSV/
│  │  ├─ final_comparison/   # Tablas maestras de comparacion
│  │  ├─ multilingual_transfer/
│  │  ├─ language_shift/
│  │  └─ pronunciation_audit/
│  ├─ attention_viz/
│  └─ checkpoints/
└─ reports/
   └─ thesis_chapter4/       # Figuras finales del capitulo 4
```

## Vision general

La investigacion sigue dos preguntas distintas:

1. **Hasta donde puede llegar el modelo principal en el dominio cerrado checo.**
2. **Que tan bien se conserva la señal patologica cuando el problema cambia de idioma y protocolo de habla.**

Por eso el repositorio mantiene dos modelos finales y no uno solo:

- El primero prioriza desempeno y sensibilidad dentro de LANNA.
- El segundo prioriza estabilidad de generalizacion entre Czech y English con una cabeza simple y explicable.

## Modelo 1: V3 LayerMix (Czech)

### Objetivo

`V3 LayerMix` es el modelo final para diagnostico experimental en el dominio cerrado checo. Su objetivo es clasificar registros `Healthy` y `Patients` del corpus LANNA usando secuencias HuBERT y un clasificador secuencial con atencion.

### Dataset y alcance

- Dataset: LANNA Czech
- Sujetos: 98
- Muestras: 2975
- Distribucion: 54 `Patients`, 44 `Healthy`
- Tarea: clasificacion binaria en dominio cerrado

### Flujo completo

```text
WAV (LANNA)
-> preprocessing.py
-> HuBERT BASE
-> secuencias .pt por muestra
-> SequenceV3Dataset
-> AttentionBiLSTMClassifierV3
-> score por muestra
-> agregacion por sujeto
-> metricas sample-level y subject-level
```

### Proceso detallado

#### 1. Preprocesamiento de audio

El flujo de audio vive principalmente en `research/src/preprocessing.py`.

Pasos:
- carga del audio y conversion a mono
- remuestreo a `16 kHz`
- filtro Butterworth pasa-altos a `100 Hz`
- normalizacion RMS con perfil `multilingual_light`
- control de pico maximo posterior a RMS
- recorte o padding cuando el experimento lo requiere

Parametros relevantes:
- `target_sr=16000`
- `highpass_hz=100`
- `target_rms=0.08`
- `max_peak_after_rms=0.99`

La decision importante aqui es que el modelo principal no usa un preprocesamiento agresivo de limpieza semantica, sino una normalizacion ligera que preserve la dinamica temporal de la pronunciacion.

#### 2. Extraccion de representaciones HuBERT

La extraccion secuencial vive en `research/src/sequence_model.py`.

Pipeline:
- se usa `torchaudio.pipelines.HUBERT_BASE`
- HuBERT produce 12 capas transformer con dimension oculta `768`
- para V3 se pueden guardar todas las capas por muestra en tensores `.pt`
- cada muestra queda representada como una secuencia temporal `(layers, time, 768)` o `(time, 768)`

El punto clave del modelo V3 es que no fija de antemano una sola capa como representacion final. En lugar de eso, aprende una mezcla escalar de capas.

#### 3. Mezcla de capas (`LayerMix`)

Cuando `use_layer_mix=true`, el clasificador aplica un `ScalarMix` aprendido sobre las 12 capas HuBERT.

Esto hace que el modelo aprenda una combinacion ponderada de capas bajas, medias y altas, en vez de depender solo de la ultima capa. En la practica, esta fue la mejora estructural que separo al V3 LayerMix de V3 control.

#### 4. Clasificador secuencial

La arquitectura principal es `AttentionBiLSTMClassifierV3`.

Flujo interno:
- `LayerNorm(768)` sobre la entrada
- `BiLSTM` bidireccional de 2 capas
- `hidden=256` por direccion
- salida temporal de dimension `512`
- mecanismo de atencion single-head
- vector de contexto final
- bloque MLP `512 -> 128 -> 1`
- `dropout=0.3`

Arquitectura resumida:

```text
HuBERT sequence
-> LayerMix
-> LayerNorm
-> BiLSTM x2
-> Attention pooling
-> LayerNorm
-> Dropout
-> Linear(512,128)
-> ReLU
-> Dropout
-> Linear(128,1)
```

#### 5. Funcion de perdida

El entrenamiento de `train_optimized_v3.py` usa una perdida compuesta:

1. `BCEWithLogitsLoss`
2. `Supervised Contrastive Loss`
3. Penalidad opcional por entropia de atencion

Configuracion principal:
- `pos_weight=1.4`
- `lambda_contrastive=0.1`
- `temperature=0.07`

Interpretacion:
- `BCE` optimiza la clasificacion binaria directa.
- la parte contrastiva fuerza que ejemplos de la misma clase queden mas cerca en el espacio latente.
- la penalidad de entropia, cuando se usa, empuja la atencion a ser mas focalizada.

#### 6. Entrenamiento

Script principal:
- `research/scripts/train_optimized_v3.py`

Configuracion:
- `epochs=30`
- optimizador `Adam`
- `lr=0.001`
- `weight_decay=0.0001`
- clipping de gradiente
- `10-fold StratifiedGroupKFold`
- agrupacion por `subject_id`

El agrupamiento por sujeto es una decision critica: evita fuga de identidad entre train y validation.

#### 7. Seleccion de umbral

Por fold, el script busca el mejor umbral sobre validacion en un rango discreto y conserva el que maximiza `recall`.

Esto refleja una prioridad clinica explicita: es preferible sostener alta sensibilidad y luego corregir precision, antes que perder pacientes.

#### 8. Agregacion por sujeto

Despues de predecir por muestra:
- se agrupan las muestras por `subject_id`
- se promedian los scores `score_patients`
- se produce una prediccion final por sujeto

Por eso el modelo reporta dos vistas:
- **sample-level**: rendimiento por muestra de audio
- **subject-level**: rendimiento por participante

### Scripts principales

- Entrenamiento: `research/scripts/train_optimized_v3.py`
- Evaluacion interna adicional con cabezas sobre embeddings: `research/scripts/evaluate_internal_cz.py`
- Auditoria de confounds: `research/scripts/audit_cz_recording_confounds.py`
- Auditoria de pronunciacion: `research/scripts/audit_pronunciation_structure.py`
- Figuras finales: `research/scripts/generate_thesis_final_model_figures.py`

### Resultados oficiales

- Metricas: `research/research_results/CSV/metrics_v3_layermix.json`
- Predicciones: `research/research_results/CSV/predictions_v3_layermix.csv`
- Comparativa maestra: `research/research_results/CSV/final_comparison/master_model_comparison.csv`
- Tabla de decisiones: `research/research_results/CSV/final_comparison/architectural_decision_table.csv`

### Hallazgos principales

#### 1. Metricas finales de V3 LayerMix

| Nivel | Accuracy | Balanced Accuracy | Precision | Recall | F1 | AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Sample | 0.8948 | 0.8783 | 0.8518 | 0.9889 | 0.9152 | 0.9451 |
| Subject | 0.9490 | 0.9432 | 0.9153 | 1.0000 | 0.9558 | 0.9739 |

Lectura:
- el modelo alcanza muy alta sensibilidad en ambos niveles
- a nivel sujeto no pierde ningun `Patient`
- el AUC subject-level de `0.9739` muestra una separacion muy fuerte entre clases

#### 2. Mejora respecto al control V3 sin LayerMix

Comparando `V3_control` vs `V3_LayerMix` en `master_model_comparison.csv`:

| Metrica | V3 control | V3 LayerMix | Delta |
| --- | ---: | ---: | ---: |
| Sample Accuracy | 0.8316 | 0.8948 | +0.0632 |
| Sample BalAcc | 0.8089 | 0.8783 | +0.0694 |
| Sample AUC | 0.9033 | 0.9451 | +0.0418 |
| Subject BalAcc | 0.8977 | 0.9432 | +0.0455 |
| Subject AUC | 0.9621 | 0.9739 | +0.0118 |

Hallazgo: el `LayerMix` no fue un detalle cosmetico. Fue una mejora estructural real que estabilizo la representacion y subio tanto precision como balanced accuracy.

#### 3. Analisis de confounds de grabacion

La auditoria vive en `audit_cz_recording_confounds.py` y prueba si diferencias de grabacion, duracion o energia podian explicar parte del rendimiento.

Se evaluaron cuatro familias de proxies:
- `task_type_only`
- `duration_only`
- `recording_proxy_only`
- `recording_proxy_plus_task`

Hallazgos clave:
- `duration_only` alcanza aproximadamente `subject_balanced_accuracy=0.7837`
- `recording_proxy_only` alcanza aproximadamente `subject_balanced_accuracy=0.9133`
- `task_type_only` esta cerca del azar

Interpretacion:
- la duracion es un confound real y no debe ignorarse
- existen diferencias de condicion de grabacion suficientemente fuertes como para producir una senal predictiva importante
- por eso el proyecto no se conforma con metricas de clasificacion; tambien audita si la senal puede venir de factores espurios

Este hallazgo es importante para el README porque explica por que se introdujeron auditorias y variantes `anti_confound` dentro del flujo experimental.

#### 4. Analisis por pronunciacion

La auditoria de pronunciacion organiza las muestras por:
- `task_group`
- `syllable_group`
- `articulatory_complexity`

El objetivo es medir en que tipo de pronunciaciones el modelo mejora o falla.

Hallazgos principales:
- no todas las tareas de pronunciacion aportan la misma informacion discriminativa
- los grupos de mayor complejidad articulatoria tienden a mostrar diferencias mas utiles entre `Healthy` y `Patients`
- el LayerMix mejora sobre el baseline de manera no uniforme: algunas familias de pronunciacion concentran la mayor ganancia
- el analisis de errores por grupo permite identificar donde aparecen falsos positivos y falsos negativos dominantes

Archivos asociados:
- `research/research_results/CSV/pronunciation_audit/`
- `research/research_results/attention_viz/pronunciation_audit/`

#### 5. Prioridad clinica: recall por sujeto

Uno de los resultados mas solidos del modelo es:

- `subject_recall = 1.0000`

En terminos practicos, dentro de esta evaluacion cerrada, todos los sujetos `Patients` fueron detectados. El costo de esta estrategia es que el problema pasa a ser mejorar precision sin sacrificar sensibilidad.

#### 6. Nota sobre la variante refinada del capitulo 4

Las figuras finales del capitulo 4 (`figura_13` a `figura_16`) describen una variante refinada posterior de V3 LayerMix basada en `hard negatives`, con metricas mejores que las de `metrics_v3_layermix.json`.

Esa variante aparece en:
- `research/reports/thesis_chapter4/capitulo_4_tablas_y_figuras.md`
- `research/scripts/generate_thesis_final_model_figures.py`

Pero sus archivos intermedios de metricas y predicciones (`metrics_v3_layermix_hardneg_hnweight2x.json`, `predictions_v3_layermix_hardneg_hnweight2x.csv`) no estan versionados actualmente en este repo. Por eso la referencia operativa estable dentro del repositorio sigue siendo `metrics_v3_layermix.json`.

### Reproduccion

Instalar dependencias:

```bash
pip install -r research/requirements.txt
```

Entrenar:

```bash
python research/scripts/train_optimized_v3.py
```

Evaluar internamente:

```bash
python research/scripts/evaluate_internal_cz.py
```

Auditar confounds:

```bash
python research/scripts/audit_cz_recording_confounds.py
```

Auditar pronunciacion:

```bash
python research/scripts/audit_pronunciation_structure.py
```

## Modelo 2: Unified Multilingual (`logreg_pca` + `centroid_shift`)

### Objetivo

El segundo modelo final no intenta reemplazar al V3 LayerMix en Czech. Su objetivo es distinto: construir una solucion unificada y ligera que preserve senal patologica a traves de dos idiomas y dos protocolos de habla distintos.

Se usa para responder: **si entrenamos una cabeza simple sobre representaciones multilingues, cual es la mejor forma de sostener desempeno simultaneo en Czech y English.**

### Dataset y alcance

- Czech: 98 sujetos, 44 `Healthy`, 54 `Patients`
- English: 32 sujetos, 20 `Healthy`, 12 `Patients`
- Flujo de secuencias: manifests naturales con preprocesamiento `multilingual_light`
- Unidad final de prediccion: sujeto

### Flujo completo

```text
Secuencias HuBERT CZ (.pt) ---\
                              -> feature engineering temporal
Secuencias HuBERT EN (.pt) ---/
                              -> pooling por sujeto
                              -> alineacion centroid_shift
                              -> StandardScaler
                              -> PCA
                              -> LogisticRegression
                              -> evaluacion bidireccional CZ y EN
```

### Proceso detallado

#### 1. Carga de secuencias HuBERT

Scripts principales:
- `research/scripts/benchmark_multilingual_cz_en_heads.py`
- `research/scripts/evaluate_multilingual_bidirectional.py`

Manifests por defecto:
- `hubert_sequences_natural_multilingual_light.csv` para Czech
- `hubert_sequences_en_test_v2.csv` para English

Cada fila apunta a un tensor `.pt` con una secuencia HuBERT. Si el tensor es 3D, el pipeline puede usar:
- `last`
- `mean_layers`

El modelo final usa `layer_policy=last`.

#### 2. Ingenieria de caracteristicas por secuencia

Cada secuencia temporal se resume en un vector mas compacto usando seis bloques estadisticos:

1. media temporal
2. desviacion estandar temporal
3. media del primer cuarto temporal
4. media del ultimo cuarto temporal
5. diferencia `first_25 - last_25`
6. pendiente temporal por dimension (`slope`)

Esto convierte una secuencia variable en un embedding fijo que aun conserva informacion de dinamica temporal gruesa.

La decision aqui es importante: este modelo no usa una red secuencial adicional. En lugar de eso, transfiere la estructura temporal a estadisticos explicables y luego usa una cabeza clasica.

#### 3. Pooling por sujeto

Cada sujeto puede tener varias muestras en Czech. Por eso el pipeline:
- agrupa por `(subject_id, label)`
- apila embeddings por muestra
- promedia todos los embeddings del sujeto

En English, el conjunto es mucho mas pequeno y suele comportarse como una muestra por sujeto. Aun asi, el pipeline mantiene la misma unidad final de evaluacion: `subject-level`.

#### 4. Alineacion entre idiomas

Se probaron tres estrategias dentro de la linea experimental:
- `raw`
- `centroid_shift`
- `coral_shift`

El modelo final unificado conserva **centroid_shift**.

##### `raw`

No transforma el embedding. La hipotesis es que la combinacion `StandardScaler + PCA` ya puede eliminar suficiente varianza de idioma sin deformar la geometria patologica.

##### `centroid_shift`

Desplaza el centroide `Healthy` del idioma fuente hacia el centroide `Healthy` del idioma objetivo.

Idea:
- restar el centroide `Healthy` de source
- sumar el centroide `Healthy` de target

##### `coral_shift`

Ademas del centrado, intenta emparejar varianzas por dimension. Fue util como prueba de language shift, pero no quedo como solucion final.

#### 5. Cabeza de clasificacion

El benchmark comparo cinco familias de cabezas:
- `logreg`
- `linear_svm`
- `logreg_pca`
- `mlp_small`
- `mlp_medium`

El ganador final fue:
- `logreg_pca`
- alineacion `centroid_shift`

Pipeline final:

```text
subject embedding
-> StandardScaler
-> PCA
-> LogisticRegression(class_weight='balanced')
```

#### 6. Evaluacion bidireccional

La evaluacion final prueba el mismo modelo sobre ambos dominios:
- metricas en `target_english`
- metricas en `target_czech`

Esto es clave porque la decision final no se tomo solo por English. Se busco una configuracion razonable para ambos idiomas a la vez.

### Scripts principales

- Benchmark de cabezas: `research/scripts/benchmark_multilingual_cz_en_heads.py`
- Evaluacion bidireccional: `research/scripts/evaluate_multilingual_bidirectional.py`
- Auditoria de language shift: `research/scripts/evaluate_language_shift_transfer.py`
- Niveles de riesgo por shift: `research/scripts/build_language_shift_risk_levels.py`

### Resultados oficiales

- Metricas bidireccionales: `research/research_results/CSV/multilingual_transfer/multilingual_bidirectional_logreg_pca_centroid.json`
- Predicciones bidireccionales: `research/research_results/CSV/multilingual_transfer/multilingual_bidirectional_logreg_pca_centroid_predictions.csv`
- Benchmark CZ+EN: `research/research_results/CSV/multilingual_transfer/multilingual_cz_en_heads_natural.json`
- Reporte de shift: `research/research_results/CSV/language_shift/language_shift_transfer_report.json`

### Hallazgos principales

#### 1. Metricas finales del modelo unificado

| Dominio objetivo | Accuracy | Balanced Accuracy | Precision | Recall | F1 | AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| English | 0.8542 | 0.8542 | 0.7931 | 0.9583 | 0.8679 | 0.9323 |
| Czech | 0.9184 | 0.9175 | 0.9259 | 0.9259 | 0.9259 | 0.9710 |

Lectura:
- English conserva un AUC alto pese al tamano reducido del dataset
- Czech mantiene un rendimiento muy competitivo sin usar una arquitectura secuencial pesada
- el modelo logra un compromiso estable en ambos idiomas, que era precisamente el objetivo del experimento

#### 2. Hallazgo central del benchmark

En `multilingual_cz_en_heads_natural.json`, el ranking final actualizado con `48` sujetos ingleses favorece a `logreg_pca + centroid_shift`.

Razon tecnica:
- mejor `balanced_accuracy` en English entre las configuraciones comparadas
- mejor `AUC` en English
- mejor `recall` en English sin deteriorar Czech en la evaluacion bidireccional
- mejora ligera tambien en `balanced_accuracy` Czech frente a `raw`
- menor complejidad y mayor interpretabilidad que un `MLP`

En otras palabras, el experimento mostro que una cabeza lineal bien regularizada sobre una proyeccion PCA era mas estable que arquitecturas mas complejas.

#### 3. Magnitud del language shift

El reporte `language_shift_transfer_report.json` muestra un hallazgo decisivo:

- norma del centroide `Healthy` en Czech: aproximadamente `20.32`
- norma del centroide `Healthy` en English: aproximadamente `13.72`
- distancia entre centroides: **`15.32`**

Interpretacion:
- el cambio de idioma y protocolo no es pequeno ni superficial
- la poblacion `Healthy` no ocupa la misma region del espacio embedding en Czech y English
- por eso no era razonable esperar que un clasificador entrenado en Czech funcionara en English sin estudiar alineacion o compresion de espacio

#### 4. Baseline English-only casi al azar

El experimento English-only LOSO reporta aproximadamente:
- `balanced_accuracy = 0.6042`
- `auc = 0.6076`

Hallazgo:
- el dataset English por si solo sigue siendo limitado para sostener un clasificador plenamente robusto
- la mejora en English aparece cuando se aprovecha informacion estructural aprendida desde Czech

Esto justifica toda la linea de transferencia multilingue.

#### 5. Por que el modelo final paso de `raw` a `centroid_shift`

La version anterior del proyecto habia dejado `logreg_pca + raw` como mejor compromiso. Sin embargo, al actualizar el dominio ingles a `48` sujetos (`24 Healthy`, `24 Patients`) y recalcular tanto el benchmark como la evaluacion bidireccional, la evidencia cambio.

Comparacion bidireccional actualizada:

- `logreg_pca + raw`
  - English: BalAcc `0.7917`, AUC `0.9253`
  - Czech: BalAcc `0.9104`, AUC `0.9735`
- `logreg_pca + centroid_shift`
  - English: BalAcc `0.8542`, AUC `0.9323`
  - Czech: BalAcc `0.9175`, AUC `0.9710`

Lectura:

1. **`centroid_shift` mejora claramente English.**
   La ganancia en English no es marginal: suben `accuracy`, `balanced_accuracy`, `precision`, `recall`, `f1` y `auc`.

2. **`centroid_shift` tambien mejora ligeramente Czech en balanced accuracy.**
   Aunque el `AUC` checo baja de forma minima, el compromiso global CZ+EN queda mejor.

3. **La nueva decision ya no depende solo del benchmark English.**
   Tambien fue validada con evaluacion bidireccional completa usando los mismos `98` sujetos checos y `48` sujetos ingleses.

La conclusion actualizada es simple: con el dataset ingles ampliado, `centroid_shift` supera a `raw` como solucion unificada final.

#### 6. Asimetria Czech-English

El analisis bidireccional muestra que la transferencia no es simetrica.

Motivos:
- Czech tiene mas sujetos y mas muestras
- English usa una tarea narrativa distinta
- el shift de protocolo modifica la distribucion embedding incluso dentro de la clase `Healthy`

Consecuencia:
- la transferencia `CZ -> EN` es viable, pero depende fuertemente de la cabeza y del tratamiento del espacio latente
- el objetivo realista no era obtener identidad de metricas entre idiomas, sino una solucion unificada razonable y estable

### Reproduccion

Instalar dependencias:

```bash
pip install -r research/requirements.txt
```

Ejecutar benchmark de cabezas:

```bash
python research/scripts/benchmark_multilingual_cz_en_heads.py
```

Ejecutar evaluacion bidireccional:

```bash
python research/scripts/evaluate_multilingual_bidirectional.py
```

Ejecutar auditoria de language shift:

```bash
python research/scripts/evaluate_language_shift_transfer.py
```

## Figuras de tesis y reportes

La salida final del capitulo 4 esta en `research/reports/thesis_chapter4/`.

Archivos relevantes:
- `figura_13_matriz_confusion_modelo_final.png`
- `figura_14_curva_roc_modelo_final.png`
- `figura_15_metricas_sample_level_modelo_final.png`
- `figura_16_metricas_subject_level_modelo_final.png`
- `figura_6_marcadores_inferidos.png`
- `figura_7_pronunciacion_marcadores.png`
- `figura_8_preprocesamiento_comparacion.png`
- `figura_9_heatmap_tp_patient.png`
- `figura_10_heatmap_fp_healthy.png`
- `figura_11_heatmap_consonante_dificil.png`

El documento `research/reports/thesis_chapter4/capitulo_4_tablas_y_figuras.md` resume textos, tablas y notas para la integracion en tesis.

## Referencias rapidas

### V3 LayerMix

- `research/src/preprocessing.py`
- `research/src/sequence_model.py`
- `research/scripts/train_optimized_v3.py`
- `research/scripts/evaluate_internal_cz.py`
- `research/scripts/audit_cz_recording_confounds.py`
- `research/scripts/audit_pronunciation_structure.py`

### Unified Multilingual

- `research/scripts/benchmark_multilingual_cz_en_heads.py`
- `research/scripts/evaluate_multilingual_bidirectional.py`
- `research/scripts/evaluate_language_shift_transfer.py`
- `research/research_results/CSV/multilingual_transfer/`
- `research/research_results/CSV/language_shift/`

## Notas finales

- `research/research_results/mlruns/` se usa solo cuando se activa seguimiento con MLflow.
- Los pesos binarios grandes y artefactos locales no se versionan aqui.
- La carpeta `research/research_history/` conserva material historico, pero la referencia activa para modelos finales, scripts vigentes y resultados finales es este README.
- El README raiz del repositorio funciona como resumen ejecutivo; este archivo es la documentacion tecnica de investigacion.
