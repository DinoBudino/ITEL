# PRONUNCIATION AUDIT: Alineacion con Kotarba (2020)

**Objetivo:** Validar sin entrenamientos nuevos si la hipotesis clinica de Kotarba
(relevancia de la pronunciacion en TEL) es compatible con los resultados de nuestros
modelos existentes.

**Fecha:** Mayo 2026

**Metodo:** Analisis de metricas, errores y scores de 3 modelos sobre el corpus LANNA
descompuesto por subtarea fonetica.

---

## 1. Taxonomia del Corpus Checo

El corpus curado (`cz_main_cleaned.csv`, 2975 muestras, 98 sujetos) tiene esta estructura
de pronunciacion:

| Grupo silabico | n muestras | n Healthy | n Patients |
|---|---:|---:|---:|
| consonant | 767 | 305 | 462 |
| word_1syll (1 silaba) | 830 | 395 | 435 |
| word_2syll (2 silabas) | 489 | 192 | 297 |
| word_3syll (3 silabas) | 395 | 170 | 225 |
| word_4syll (4 silabas) | 296 | 113 | 183 |
| word_difficult (VSL) | 198 | 91 | 107 |

**No hay muestras de vocales** en el dataset curado. Solo consonantes y palabras por
complejidad silabica creciente.

**Complejidad articulatoria derivada:**
- `low`: word_1syll (830)
- `medium`: consonant + word_2syll (1256)
- `high`: word_3syll + word_4syll + word_difficult (889)

---

## 2. Rendimiento por Grupo de Pronunciacion (V3 LayerMix)

| Grupo | Bal Acc | Recall | AUC | Precision | n |
|---|---:|---:|---:|---:|---:|
| word_3syll | **0.898** | 0.995 | 0.945 | 0.862 | 395 |
| word_4syll | **0.894** | 1.000 | 0.962 | 0.854 | 296 |
| word_difficult | 0.882 | 0.991 | 0.969 | 0.845 | 198 |
| word_2syll | 0.851 | 1.000 | 0.929 | 0.807 | 489 |
| word_1syll | 0.845 | 0.992 | 0.946 | 0.820 | 830 |
| consonant | 0.813 | 1.000 | 0.942 | 0.802 | 767 |

**Hallazgos:**
1. Las palabras de 3-4 silabas son las mas discriminativas (bal_acc ~0.89-0.90).
2. Las consonantes aisladas son las peores (bal_acc = 0.813), a pesar de tener recall=1.0.
3. La precision es sistematicamente menor en todos los grupos que el recall, indicando
   que el modelo es conservador con los falsos positivos.

---

## 3. Impacto Diferencial de LayerMix por Pronunciacion

Delta de balanced accuracy (V3 LayerMix vs V3 control) por grupo:

| Grupo | Delta LayerMix | Interpretacion |
|---|---:|---|
| word_2syll | **+0.1404** | Mayor beneficio |
| consonant | **+0.1365** | Muy alto beneficio |
| word_3syll | +0.1267 | Beneficio alto |
| word_1syll | +0.1195 | Beneficio moderado |
| word_4syll | +0.0901 | Beneficio moderado |
| word_difficult | +0.0693 | Menor beneficio |

**Interpretacion:**
1. LayerMix mejora TODOS los grupos. No empeora ninguno.
2. El mayor beneficio esta en **consonantes y palabras de 2-3 silabas**, no en las mas
   complejas.
3. Las palabras dificiles (VSL) y de 4 silabas reciben el menor boost de LayerMix.
4. Esto sugiere que **las capas extra de HuBERT capturan mejor la microestructura
   fonetica** (consonantes, palabras simples) que los patrones de palabras complejas,
   donde la ultima capa ya funciona razonablemente bien.

---

## 4. Perfil de Errores

### Sujetos criticos

| Sujeto | Modelo | Error Rate | FP | Notas |
|---|---:|---:|---:|---|
| H26 | Todos | 1.00 | 24/24 | 100% FP, el mas dificil |
| H27 | V3_LayerMix | 0.93 | 25/27 | Mejora leve con LayerMix |
| H28 | V3_LayerMix | 0.94 | 31/33 | Mejora leve con LayerMix |
| H29 | V3_LayerMix | 1.00 | 27/27 | Sin mejora con LayerMix |

Los sujetos H26-H29 siguen siendo los mas problematicos. LayerMix reduce FP en H27 y H28
pero no en H26 ni H29.

### Distribucion de FP por grupo silabico (V3 LayerMix)

La mayoria de FP ocurren en `word_1syll` y `consonant` (los grupos con mayor volumen de
muestras). La tasa de FP normalizada por grupo muestra que `consonant` tiene la mayor
proporcion de errores.

---

## 5. Alineacion con Kotarba

### Lo que Kotarba afirma
- El TEL se manifiesta fuertemente en la pronunciacion
- Una ResNet-18 sobre espectrogramas log-power alcanza 99.48% en LANNA
- La senal esta en patrones espectro-temporales locales

### Lo que nuestros datos muestran
1. **El rendimiento SI varia por tipo de tarea fonetica.** Las palabras de 3-4 silabas
   separan mejor que las consonantes aisladas. Esto es compatible con Kotarba: hay
   estructura de pronunciacion en los datos.

2. **Pero el patron es el opuesto al esperado.** Kotarba implicaria que las tareas mas
   complejas articulatoriamente deberian ser las mas discriminativas. Nuestros datos
   muestran que:
   - Las palabras complejas (4 silabas, VSL) ya se separan bien incluso sin LayerMix
   - Las consonantes y palabras simples son las que mas se benefician de features extra

3. **LayerMix no es una ResNet, y aun asi mejora.** El hecho de que una simple suma
   pesada de capas ya de +5-14pp de balanced accuracy en ciertos grupos sugiere que
   la senal fonetica existe y es accesible sin CNN.

### Conclusion sobre Kotarba

**La hipotesis de Kotarba es PARCIALMENTE compatible con nuestros datos.**

- Compatible en: la pronunciacion importa, y el rendimiento no es uniforme entre tareas
- No compatible en: la complejidad articulatoria NO es el principal driver de
  discriminabilidad en nuestro caso

---

## 6. Recomendacion Final

### GO / NO-GO para benchmark ResNet-18

**Veredicto: NO-GO inmediato. Primero explotar subtareas con arquitectura actual.**

Justificacion:
1. La variabilidad por pronunciacion es real pero no tan dramatica como para justificar
   un cambio completo de arquitectura.
2. LayerMix ya captura parte del beneficio que una CNN daria (features multi-nivel), y
   lo hace sin costo de reentrenamiento de backbone.
3. Las consonantes aisladas son el punto debil actual. Una estrategia barata de probar
   antes que ResNet es:
   - modelo especializado solo en consonantes
   - o late fusion de scores por tipo de tarea
   - o data augmentation focalizado en ese grupo

### Siguientes pasos recomendados

1. **Cerrar mejora B (embedding noise) en checo** -- podria ayudar especialmente en
   consonantes al forzar invariancia a variaciones de pronunciacion.
2. **Analisis de atencion por tipo de tarea** -- ver si el modelo atiende diferente
   en consonantes vs palabras complejas.
3. **Modelo especializado por subtarea** -- entrenar un clasificador solo en consonantes
   y otro solo en palabras, fusionar a nivel sujeto.
4. **Solo si nada de esto sube consonantes por encima de 0.85 bal_acc**, considerar
   benchmark ResNet-18 como exploracion arquitectonica.

---

## 7. Artefactos Generados

| Archivo | Descripcion |
|---|---|
| `research/scripts/audit_pronunciation_structure.py` | Script completo de auditoria |
| `research/research_results/CSV/pronunciation_audit/pronunciation_master_table.csv` | Taxonomia del corpus |
| `research/research_results/CSV/pronunciation_audit/metrics_by_task_type.csv` | Metricas por task_type |
| `research/research_results/CSV/pronunciation_audit/metrics_by_syllable_group.csv` | Metricas por grupo silabico |
| `research/research_results/CSV/pronunciation_audit/metrics_by_articulatory_complexity.csv` | Metricas por complejidad |
| `research/research_results/CSV/pronunciation_audit/model_delta_by_pronunciation_group.csv` | Deltas entre modelos |
| `research/research_results/CSV/pronunciation_audit/error_distribution_by_group.csv` | Distribucion de errores |
| `research/research_results/CSV/pronunciation_audit/subject_error_report.csv` | Errores por sujeto |
| `research/research_results/attention_viz/pronunciation_audit/score_distribution_by_task_type.png` | Grafico scores |
| `research/research_results/attention_viz/pronunciation_audit/score_distribution_by_syllable_group.png` | Grafico scores |
| `research/research_results/attention_viz/pronunciation_audit/error_profile_by_group.png` | Grafico errores |
| `PRONUNCIATION_AUDIT_KOTARBA_ALIGNMENT.md` | Este documento |
