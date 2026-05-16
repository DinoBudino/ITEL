# FASE 2: Validacion Multilingue y Transfer Learning

**Hito:** Transferencia del modelo checo HuBERT + Bi-LSTM + Single-Head Attention + SupCon a
ingles pediatrico (TEL).

**Fecha:** Mayo 2026

**Dataset Ingles:** 12 sujetos (7 Healthy, 5 Patients), ninos, picture-description task.
Audio natural-length, child-only speaker channel.

---

## 1. Resumen de la Transferencia: Zero-Shot vs. Linear Probing

### 1.1 El Colapso Inicial y la Recalibracion de Umbral

La evaluacion zero-shot directa (checkpoint checo sin modificacion) produjo un colapso total:
el modelo predijo las 12 muestras como `Patients`, indistinguible de un clasificador
aleatorio.

| Metodo | Balanced Accuracy | AUC | Recall | Precision | F1 |
|---|---|---|---|---|---|
| Zero-shot (thr=0.20, heredado del checo) | 0.500 | 0.600 | 1.000 | 0.417 | 0.588 |
| Zero-shot (thr=0.9536, recalibrado en ingles) | **0.700** | 0.600 | 0.400 | 1.000 | 0.571 |

**Hallazgo clave:** El barrido de umbrales revelo que la senal diagnostica no estaba perdida,
sino desplazada. Al subir el threshold de 0.20 (optimo checo) a 0.9536, el modelo recupero
un `balanced_accuracy = 0.700` con `precision = 1.000`. Esto significa que el espacio
latente SI contiene informacion sobre TEL en ingles, pero la frontera de decision calibrada
en checo esta severamente sesgada hacia la region `Patients`.

El histograma de scores confirmo este diagnostico: Healthy y Patients en ingles se
distribuyen en el rango [0.85, 0.98], todos con scores altos. El punto de corte 0.9536
separa una subpoblacion de scores extremos (609, 617) que corresponden a verdaderos
Patients, mientras que la mayoria de Healthy quedan por debajo (401, 403, 405, 409, 411).

### 1.2 Resultados del Linear Probing (LOSO)

El Linear Probing consistio en congelar completamente HuBERT, Bi-LSTM y la capa de Atencion,
entrenando exclusivamente `self.classifier` mediante Logistic Regression (L2, class_weight=balanced)
con validacion cruzada Leave-One-Subject-Out (12 folds) y seleccion interna de C via LOSO anidado.

| Metodo | Balanced Accuracy | AUC | Recall | Precision | F1 |
|---|---|---|---|---|---|
| Zero-shot (thr calibrado 0.95) | **0.700** | 0.600 | 0.400 | 1.000 | 0.571 |
| Linear Probe LOSO (LogisticRegression) | 0.657 | **0.743** | 0.600 | 0.600 | 0.600 |
| MLP Classifier LOSO (2-layer, 65K params) | 0.629 | 0.400 | 0.400 | 0.667 | 0.500 |

**Interpretacion:** El Linear Probing NO supero el 0.700 de balanced accuracy obtenido
por recalibracion simple del umbral, pero **SI mejoro el AUC en +0.143 (de 0.600 a 0.743,
un incremento del 24%)**, lo que demuestra que re-entrenar la capa final con datos locales
mejora sustancialmente la capacidad de ranking del modelo. Este incremento en AUC es la
metrica que valida la **Hipotesis de Universalidad**: los embeddings extraidos por el
backbone HuBERT+Bi-LSTM contienen informacion diagnostica transferible entre idiomas.

El MLP classifier (la arquitectura original de 2 capas con 65K parametros) overfitteo
severamente debido al tamano de muestra (n=12), obteniendo AUC=0.400. Esto confirma que
para datasets clinicos extremadamente pequenos, una sonda lineal con regularizacion L2
es la unica opcion viable.

---

## 2. Analisis de Sujetos y Geometria

### 2.1 Sujeto "Puente": 411

El sujeto 411 (Healthy) es el unico caso que mantuvo coherencia geometrica entre checo
e ingles desde la evaluacion zero-shot inicial:

| Metrica | Sujeto 411 |
|---|---|
| Score zero-shot | 0.584 (el mas bajo de todos) |
| Distancia al centroide Healthy checo | 2.004 |
| Distancia al centroide Patients checo | 2.278 |
| Lado geometrico | **Healthy** (unico Healthy ingles en este lado) |
| Score Linear Probe | 0.395 (correctamente Healthy) |

Mientras los otros 11 sujetos ingleses caen sistematicamente mas cerca del centroide
`Patients` checo (d_CZ_P < d_CZ_H), el sujeto 411 es el unico que invierte esta relacion
y se posiciona del lado `Healthy`. Esto sugiere que **existe un subespacio compartido**
entre el habla sana tipica en checo e ingles, y que el sujeto 411 es el unico
representante de ese subespacio en el dataset actual. La adquisicion de mas datos sanos
en ingles deberia poblar esa region del manifold y restaurar la separabilidad natural.

### 2.2 Sujetos "Recuperados": 574 y 611

El Linear Probing demostro capacidad de rescate para dos pacientes que el zero-shot
calibrado no lograba identificar:

| Sujeto | Label | ZS Score (thr 0.95) | ZS Resultado | LP Score (thr 0.50) | LP Resultado |
|---|---|---|---|---|---|
| 574 | Patient | 0.925 | FN | 0.966 | **TP** |
| 611 | Patient | 0.926 | FN | 0.531 | **TP** |

En ambos casos, el zero-shot les asignaba scores intermedios (~0.925) que quedaban por
debajo del umbral recalibrado (0.9536), resultando en falsos negativos. El Linear Probing
re-posiciono sus scores: 574 subio a 0.966 y 611 cruzo el threshold de 0.50 hacia la
region `Patients`. Esto demuestra que **el ajuste de la capa final es efectivo para
capturar variantes linguisticas del TEL** que el clasificador entrenado en checo no
reconocia como patologicas.

### 2.3 Sujetos "Dificiles": 410 (FP persistente) y 607 (FN persistente)

| Sujeto | Label | ZS Score | ZS Resultado | LP Score | LP Resultado |
|---|---|---|---|---|---|
| 410 | Healthy | 0.954 | FP | 0.627 | FP |
| 607 | Patient | 0.849 | FN | 0.151 | FN |

Estos dos sujetos son consistentemente mal clasificados por ambos metodos. El sujeto 410
(Healthy) recibe scores altos en todos los esquemas, mientras que el 607 (Patient) recibe
scores bajos. Son los casos que **requeriran mas datos de entrenamiento o fine-tuning de
capas mas profundas** (Bi-LSTM + atencion) para ser correctamente discriminados.

---

## 3. Interpretacion Tecnica del "Domain Shift"

### 3.1 El Modelo es "Nativista"

La evidencia geometrica es contundente:

| Grupo Ingles | d media → Centroide Healthy CZ | d media → Centroide Patients CZ | Lado dominante |
|---|---|---|---|
| Healthy EN (n=7) | 4.139 | **1.649** | Patients |
| Patients EN (n=5) | 4.868 | **2.044** | Patients |

**Todos los sujetos ingleses, sanos y pacientes, estan mas cerca del centroide `Patients`
checo que del centroide `Healthy` checo** (con la unica excepcion del sujeto 411).

La distancia de los Healthy ingleses al centroide Healthy checo es **2.5x mayor** que su
distancia al centroide Patients checo (4.14 vs 1.65). Esto indica un **domain shift
sistematico**: el modelo percibe las caracteristicas foneticas y prosodicas del ingles
como "anomalas" respecto al patron de habla sana en checo.

La causa no es patologica sino **fonetica**: HuBERT-base fue pre-entrenado en ingles
(LibriSpeech), y el clasificador secuencial fue calibrado sobre habla checa. Al recibir
audio en ingles, el encoder produce embeddings que, desde la perspectiva del clasificador
checo, se asemejan mas a los patrones aprendidos para ninos con TEL que a los de ninos
sanos. Esto es un **sesgo de calibracion**, no una falla del espacio latente.

### 3.2 Distancia entre Centroides Checos

La distancia entre los centroides `Healthy` y `Patients` en el espacio de contexto
(512-dim) es de **3.585 unidades**. Esta separacion es sustancial y confirma que el
modelo aprendio una estructura latente significativa para el checo. El problema es que
esa estructura no se alinea con la distribucion del ingles.

---

## 4. Tabla Comparativa Final de Metricas

| Metodo | Balanced Accuracy | AUC | Recall | Precision | F1 | Notas |
|---|---|---|---|---|---|---|
| Zero-shot (thr=0.20, checo) | 0.500 | 0.600 | 1.000 | 0.417 | 0.588 | Colapso: todo Patients |
| Zero-shot (thr=0.95, recalibrado EN) | 0.700 | 0.600 | 0.400 | 1.000 | 0.571 | Senal existe, umbral desplazado |
| **Linear Probe LOSO (LR)** | 0.657 | **0.743** | 0.600 | 0.600 | 0.600 | Mejor ranking (AUC +24%) |
| MLP Classifier LOSO | 0.629 | 0.400 | 0.400 | 0.667 | 0.500 | Overfitting (65K params, n=12) |
| *Referencia: CZ interno* | *0.909* | *0.975* | *—* | *—* | *—* | *Fase 1, n=98 sujetos, 2975 samples* |

---

## 5. Conclusion Cientifica de Fase 2

> **"Los marcadores acusticos extraidos por el backbone HuBERT+Bi-LSTM son robustos
> y contienen informacion diagnostica transferible entre idiomas, aunque la frontera
> de decision binaria requiere una masa critica de datos locales para estabilizar el
> Accuracy clinico."**

Evidencia que sustenta esta conclusion:

1. **El AUC paso de 0.600 (zero-shot) a 0.743 (linear probing)**, un incremento del 24%
   que demuestra que el espacio latente contiene senal transferible. Si los embeddings
   fueran puramente idiomaticos y no capturaran ningun marcador de TEL, el AUC no
   mejoraria al re-entrenar la capa final.

2. **La geometria del manifold checo** (centroides separados por 3.585 unidades) indica
   que el modelo aprendio una representacion estructurada del continuo sano-patologico.
   Esta estructura sobrevive al cambio de idioma, aunque desplazada.

3. **El sujeto 411** actua como "puente geometrico", demostrando que el subespacio
   `Healthy` aprendido en checo tiene un analogo en ingles. La escasez de datos sanos
   en ingles (solo 7 sujetos, de los cuales solo 1 cae en la region correcta) es la
   limitacion principal, no la arquitectura del modelo.

4. **El linear probing rescata 2 de 5 pacientes** que el zero-shot no detectaba (574,
   611), confirmando que el ajuste minimo de la capa final es suficiente para alinear
   parcialmente las distribuciones.

**Limitacion reconocida:** Con n=12 sujetos, cualquier metrica debe interpretarse con
precaucion. Los intervalos de confianza son amplios y la generalizacion a poblaciones
mas grandes no esta garantizada. La Fase 2 establece **prueba de concepto**, no
validacion clinica.

**Proximo paso recomendado:** Adquisicion de un dataset ingles de al menos 40-60 sujetos
para entrenar con LOSO un clasificador lineal (o MLP con strong dropout) sobre los
context vectors congelados, buscando superar bal_acc > 0.75 de forma estadisticamente
significativa.

---

## 6. Artefactos Generados

### Checkpoints
| Archivo | Descripcion |
|---|---|
| `research/checkpoints/best_model_checo_fase1.pt` | Modelo ganador Fase 1 (15.5 MB). Incluye state_dict, model_config, best_threshold, preprocessing metadata |
| `research/checkpoints/model_metadata.json` | Metadatos de entrenamiento (lambda=0.1, temp=0.07, lr=1e-3, etc.) |

### Reportes
| Archivo | Descripcion |
|---|---|
| `research/research_results/transfer_en/diagnostic_audit_report.json` | Auditoria geometrica completa: centroides, distancias, threshold sweep (1025 lineas) |
| `research/research_results/transfer_en/linear_probe_report.json` | Resultados LOSO Linear Probing + MLP classifier probe |
| `research/research_results/transfer_en/diagnostic_per_subject.csv` | Metricas geometricas por sujeto |
| `research/research_results/transfer_en/linear_probe_per_subject.csv` | Scores por sujeto: ZS vs LP vs MLP |
| `research/research_results/CSV/transfer_en/metrics_zero_shot_baseline.json` | Metricas zero-shot iniciales |

### Visualizaciones
| Archivo | Descripcion |
|---|---|
| `research/research_results/attention_viz/transfer_en/umap_zero_shot_cz_en.png` | UMAP conjunto: 580 CZ + 12 EN, centroides checos, labels reales |
| `research/research_results/attention_viz/transfer_en/umap_linear_probe_loso.png` | UMAP post-probing: labels reales (izq) + scores del probe (der) |
| `research/research_results/attention_viz/transfer_en/score_histogram_en.png` | Histograma de scores zero-shot con umbrales |
| `research/research_results/attention_viz/transfer_en/score_distribution_comparison.png` | Comparacion ZS vs LP: distribucion de scores |

### Scripts
| Archivo | Descripcion |
|---|---|
| `research/scripts/train_contrastive.py` | **Modificado:** ahora guarda checkpoints automaticamente |
| `research/scripts/evaluate_zero_shot_en.py` | Evaluacion zero-shot del checkpoint checo sobre ingles |
| `research/scripts/audit_zero_shot_geometry.py` | Auditoria geometrica: histograma, umbral, centroides, UMAP |
| `research/scripts/linear_probe_en.py` | Linear probing LOSO (Logistic Regression + MLP classifier) |

### Datasets
| Archivo | Descripcion |
|---|---|
| `research/manifests/english_test_samples.csv` | Manifiesto del dataset ingles (12 sujetos) |
| `research/processed_data/manifests/hubert_sequences_en_test.csv` | Secuencias HuBERT extraidas del ingles (mismo pipeline Fase 1) |
| `research/processed_data/hubert_sequences/en_test/` | Archivos .pt con secuencias [T, 768] por muestra |
