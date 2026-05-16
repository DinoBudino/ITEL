# Capitulo 4 - Tablas y figuras de resultados

Este archivo reune las tablas y notas de figuras generadas para el Capitulo 4, utilizando como fuentes `metrics_v3.json`, `metrics_v3_layermix.json`, `predictions_v3.csv`, `predictions_v3_layermix.csv` y `metadata.json` del checkpoint final `v3_final/layermix`.

Ubicacion sugerida de la Figura 1: 4.2.3 Objetivo especifico 3, despues del parrafo que presenta la arquitectura V3 LayerMix y antes de la tabla de arquitectura e hiperparametros.

Figuras generadas

Figura 1.
Arquitectura del modelo V3 LayerMix

Nota. La figura muestra el diseño general del modelo propuesto, compuesto por la entrada de audio, extraccion de representaciones mediante HuBERT, fusion de capas con LayerMix, procesamiento secuencial mediante Bi-LSTM, mecanismo de atencion y clasificador binario para la salida Healthy/Patient. Elaboracion propia.

Figura 2.
Matriz de confusion del modelo V3 LayerMix

Nota. La matriz de confusion se calculo a partir de las predicciones reales del modelo V3 LayerMix: TN=899, FP=367, FN=6, TP=1703. Elaboracion propia.

Figura 3.
Curva ROC del modelo V3 LayerMix

Nota. La curva ROC se calculo con los scores reales del archivo predictions_v3_layermix.csv. AUC = 0.9451. Elaboracion propia.

Figura 4.
Comparacion grafica de metricas sample-level

Nota. Compara Accuracy, Balanced Accuracy, Precision, Recall, F1-Score y AUC-ROC entre V3 Control y V3 LayerMix. Elaboracion propia.

Figura 5.
Comparacion grafica de metricas subject-level

Nota. Compara Subject Accuracy, Subject Balanced Accuracy, Subject Precision, Subject Recall, Subject F1-Score y Subject AUC entre V3 Control y V3 LayerMix. Elaboracion propia.

Tabla 2.
Caracteristicas de las representaciones extraidas con HuBERT

| Parametro | Valor |
| --- | --- |
| Modelo extractor | HuBERT-base |
| Estado del extractor | Congelado |
| Capas extraidas | 12 |
| Dimension por capa | 768 |
| Formato de representacion | [12, T, 768] |
| Muestras totales | 2975 |
| Muestras Patient | 1709 |
| Muestras Healthy | 1266 |
| Sujetos | 98 |
| Estrategia temporal | Natural-length |

Nota. Elaboracion propia.

Tabla 3.
Arquitectura e hiperparametros del modelo V3 LayerMix

| Parametro | Valor |
| --- | --- |
| Extractor acustico | HuBERT-base congelado |
| Capas utilizadas | 12 capas de HuBERT |
| Fusion de capas | LayerMix / ScalarMix |
| Normalizacion | LayerNorm |
| Modelo secuencial | Bi-LSTM bidireccional |
| Capas Bi-LSTM | 2 |
| Tamano oculto | 256 |
| Salida Bi-LSTM | 512 dimensiones |
| Mecanismo de atencion | Single-Head Attention |
| Clasificador | Linear(512->128) -> ReLU -> Linear(128->1) |
| Funcion de perdida | BCE + Supervised Contrastive Loss |
| Optimizador | Adam |
| Learning rate | 0.001 |
| Weight decay | 1e-4 |
| Batch size | 16 |
| Epocas de entrenamiento | 30 |

Nota. Elaboracion propia.

Tabla 4.
Modelos evaluados

| Modelo | Descripcion |
| --- | --- |
| V3 Control | Utiliza unicamente la ultima capa de HuBERT como entrada para la Bi-LSTM. |
| V3 LayerMix | Utiliza una combinacion aprendible de las 12 capas de HuBERT. |

Nota. Elaboracion propia.

Tabla 5.
Rendimiento del modelo en validacion cruzada a nivel de muestra

| Metrica | V3 Control | V3 LayerMix | Diferencia |
| --- | --- | --- | --- |
| Accuracy | 0.8316 | 0.8948 | +0.0632 |
| Balanced Accuracy | 0.8089 | 0.8783 | +0.0694 |
| Precision | 0.7907 | 0.8518 | +0.0612 |
| Recall | 0.9614 | 0.9889 | +0.0275 |
| F1-Score | 0.8677 | 0.9152 | +0.0475 |
| AUC-ROC | 0.9033 | 0.9451 | +0.0418 |

Nota. Elaboracion propia.

Tabla 6.
Rendimiento del modelo en validacion cruzada a nivel de sujeto

| Metrica | V3 Control | V3 LayerMix | Diferencia |
| --- | --- | --- | --- |
| Subject Accuracy | 0.9082 | 0.9490 | +0.0408 |
| Subject Balanced Accuracy | 0.8977 | 0.9432 | +0.0455 |
| Subject Precision | 0.8571 | 0.9153 | +0.0581 |
| Subject Recall | 1.0000 | 1.0000 | +0.0000 |
| Subject F1-Score | 0.9231 | 0.9558 | +0.0327 |
| Subject AUC | 0.9621 | 0.9739 | +0.0118 |

Nota. Elaboracion propia.

Tabla 7.
Estabilidad del modelo entre folds

| Metrica | V3 Control | V3 LayerMix |
| --- | --- | --- |
| Mean Fold Balanced Accuracy | 0.7310 ± 0.1108 | 0.8591 ± 0.0680 |
| Mean Fold AUC | 0.9417 ± 0.0363 | 0.9657 ± 0.0394 |
| Mean Fold F1 | 0.8302 ± 0.0749 | 0.9001 ± 0.0595 |
| Mean Fold Recall | 0.9874 ± 0.0116 | 0.9962 ± 0.0043 |

Nota. Elaboracion propia.

Tabla 8.
Comparacion del modelo propuesto con modelos del estado del arte

| Autor / Estudio | Enfoque | Metrica principal |
| --- | --- | --- |
| Hasib et al. | DCNN | 90.47% Accuracy |
| Kaushik et al. | 2D-CNN | 99.09% Accuracy |
| Kotarba & Kotarba | ResNet-18 | 99.48% Accuracy |
| Reddy et al. | FFNN + caracteristicas glotales | 98.82% Accuracy |
| Sharma et al. | LBP + caracteristicas acusticas | 97.36% Accuracy |
| Modelo propuesto | HuBERT + LayerMix + Bi-LSTM + Attention | 89.48% Sample Accuracy / 94.90% Subject Accuracy |

Nota. Elaboracion propia.
