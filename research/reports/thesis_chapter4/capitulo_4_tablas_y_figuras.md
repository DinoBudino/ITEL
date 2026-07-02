# Capitulo 4 - Tablas, figuras y ajustes finales

Este archivo reune las tablas, notas de figuras y textos breves sugeridos para el Capitulo 4. La version actual distingue solo entre el modelo base y el modelo refinado final, dejando el analisis en ingles como complemento exploratorio.

## Figuras recomendadas

No se recomienda agregar mas figuras. Para mantener claridad, las figuras centrales del capitulo deberian ser:

| Seccion | Figura | Uso sugerido |
| --- | --- | --- |
| 4.1 | Arquitectura general del sistema | Presentar el flujo completo de la propuesta. |
| 4.2.2 | Modulo de extraccion acustica con HuBERT | Explicar la obtencion de representaciones acusticas. |
| 4.2.3 | Modelo de clasificacion secuencial | Explicar la arquitectura del clasificador. |
| 4.2.3 | Proceso de construccion del modelo final | Mostrar el paso del modelo base al modelo refinado final. |
| 4.2.3 | Rendimiento por pronunciacion | Justificar que algunas pronunciaciones aportan mas informacion discriminativa. |
| 4.2.4 | Matriz de confusion del modelo refinado final | Mostrar aciertos y errores de clasificacion. |
| 4.2.4 | Curva ROC del modelo refinado final | Mostrar capacidad discriminativa global. |
| 4.2.4 | Metricas sample-level | Comparar modelo base y modelo refinado final a nivel de muestra. |
| 4.2.4 | Metricas subject-level | Comparar modelo base y modelo refinado final a nivel de sujeto. |

Si el documento queda muy cargado, se pueden retirar figuras intermedias que repitan la arquitectura o la separacion de datos, especialmente si ya estan explicadas en los diagramas principales.

## Figuras 13-16 generadas

Los archivos finales estan en `research/reports/thesis_chapter4/`:

| Figura | Archivo | Titulo sugerido |
| --- | --- | --- |
| Figura 13 | `figura_13_matriz_confusion_modelo_final.png` | Matriz de confusion del modelo refinado final |
| Figura 14 | `figura_14_curva_roc_modelo_final.png` | Curva ROC del modelo refinado final |
| Figura 15 | `figura_15_metricas_sample_level_modelo_final.png` | Comparacion de metricas a nivel de muestra |
| Figura 16 | `figura_16_metricas_subject_level_modelo_final.png` | Comparacion de metricas a nivel de sujeto |

### Figura 13

Matriz de confusion del modelo refinado final.

Nota. La matriz de confusion muestra los aciertos y errores del modelo refinado final en la clasificacion de registros Healthy y Patients. Se obtuvieron 1187 verdaderos negativos, 79 falsos positivos, 60 falsos negativos y 1649 verdaderos positivos. Elaboracion propia.

### Figura 14

Curva ROC del modelo refinado final.

Nota. La curva ROC se calculo a partir de los puntajes de salida del modelo refinado final. El area bajo la curva fue de 0.9806, lo que evidencia una alta capacidad discriminativa entre los grupos Healthy y Patients. Elaboracion propia.

### Figura 15

Comparacion de metricas a nivel de muestra.

Nota. La figura compara el desempeno del modelo base y del modelo refinado final a nivel de muestra, considerando Accuracy, Balanced Accuracy, Precision, Recall, F1-Score y AUC-ROC. Elaboracion propia.

### Figura 16

Comparacion de metricas a nivel de sujeto.

Nota. La figura compara el desempeno del modelo base y del modelo refinado final a nivel de sujeto, considerando Accuracy, Balanced Accuracy, Precision, Recall, F1-Score y AUC. Elaboracion propia.

## Tabla de metricas sample-level

| Metrica | Modelo base | Modelo refinado final | Diferencia |
| --- | ---: | ---: | ---: |
| Accuracy | 89.48% | 94.59% | +5.11 pp |
| Balanced Accuracy | 87.83% | 94.34% | +6.50 pp |
| Precision | 85.18% | 94.64% | +9.46 pp |
| Recall | 98.89% | 96.02% | -2.87 pp |
| F1-Score | 91.52% | 95.32% | +3.80 pp |
| AUC-ROC | 94.51% | 98.06% | +3.55 pp |

Nota. Las metricas a nivel de muestra muestran que el modelo refinado final mejora el desempeno global del modelo base, especialmente en precision, balanced accuracy y AUC-ROC. Elaboracion propia.

## Tabla de metricas subject-level

| Metrica | Modelo base | Modelo refinado final | Diferencia |
| --- | ---: | ---: | ---: |
| Accuracy | 94.90% | 97.96% | +3.06 pp |
| Balanced Accuracy | 94.32% | 97.73% | +3.41 pp |
| Precision | 91.53% | 96.43% | +4.90 pp |
| Recall | 100.00% | 100.00% | 0.00 pp |
| F1-Score | 95.58% | 98.18% | +2.61 pp |
| AUC | 97.39% | 100.00% | +2.61 pp |

Nota. Las metricas a nivel de sujeto indican que el modelo refinado final mantiene una sensibilidad completa y mejora la precision general de la clasificacion por participante. Elaboracion propia.

## Tabla de comparacion con literatura

| Autor / estudio | Enfoque | Metrica principal reportada |
| --- | --- | --- |
| Hasib et al. | DCNN | 90.47% Accuracy |
| Kaushik et al. | 2D-CNN | 99.09% Accuracy |
| Kotarba & Kotarba | ResNet-18 | 99.48% Accuracy |
| Reddy et al. | FFNN + caracteristicas glotales | 98.82% Accuracy |
| Sharma et al. | LBP + caracteristicas acusticas | 97.36% Accuracy |
| Modelo refinado final | HuBERT + fusion multicapa + Bi-LSTM + atencion | 94.59% sample-level Accuracy / 97.96% subject-level Accuracy |

Nota. La comparacion debe interpretarse con cautela, debido a diferencias entre corpus, idiomas, tareas de habla, criterios de evaluacion y particiones experimentales. Elaboracion propia.

## Analisis complementario en ingles

Texto sugerido para la nota de la tabla correspondiente:

Nota. Los resultados corresponden a una evaluacion exploratoria en ingles y se presentan como analisis complementario. Debido al tamano reducido del conjunto de datos y a las diferencias de idioma y protocolo, estos resultados no sustituyen la evaluacion principal realizada sobre el corpus en espanol.

## Docimasia de hipotesis

Texto sugerido para reemplazar `No aplica`:

De acuerdo con los resultados obtenidos, se rechaza la hipotesis nula y se acepta la hipotesis general de investigacion. El modelo refinado final alcanzo un desempeno elevado tanto a nivel de muestra como a nivel de sujeto, con 94.59% de accuracy, 94.34% de balanced accuracy, 95.32% de F1-Score y 98.06% de AUC-ROC a nivel de muestra. Asimismo, a nivel de sujeto alcanzo 97.96% de accuracy, 97.73% de balanced accuracy, 98.18% de F1-Score y 100.00% de AUC.

Estos resultados evidencian que las representaciones acusticas extraidas mediante HuBERT, combinadas con un modelo secuencial de clasificacion, permiten discriminar entre registros Healthy y Patients con un rendimiento consistente. Por tanto, la evidencia experimental respalda la viabilidad de la propuesta como herramienta computacional de apoyo para la deteccion temprana del trastorno especifico del lenguaje.

## Correcciones pendientes en Word o PDF

- Actualizar el indice de tablas y figuras para que no conserve referencias antiguas a MLP, PCA o versiones previas del modelo.
- Sustituir cualquier aparicion de `MSFM` por `modelo base` o `modelo refinado final`, segun corresponda.
- Corregir el error tipografico `os resultados` por `Los resultados`.
- Revisar que el valor de Precision del estudio de ResNet coincida con la fuente real antes de dejar la tabla final.
- Mantener la denominacion `modelo refinado final` en la discusion, conclusiones y docimasia para evitar mezclar siglas innecesarias.
