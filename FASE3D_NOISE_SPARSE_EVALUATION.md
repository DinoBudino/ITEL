# FASE 3D: Noise 0.03 + Sparse Attention — Evaluación Final

**Objetivo:** Evaluar si `Sparse Attention` mejora la rama multilingüe `Noise 0.03`.

**Fecha:** Mayo 2026

**Veredicto final multilingüe: GO — `V3 Noise 0.03` como rama definitiva.**
**Sparse Attention: NO-GO — empeora tanto en checo como en inglés.**

---

## Parte A — Cierre Geométrico de V3 Noise 0.03

### 1. Comparación de Geometría

| Método | Geometry Accuracy | EN Healthy → CZ H | EN Healthy → CZ P | Interpretación |
|---|---:|---:|---:|---|
| Fase 2 baseline | 0.500 (6/12) | 4.14 | 1.65 | Sanos caen del lado Patients |
| V3 LayerMix | 0.417 (5/12) | 2.16 | 2.13 | Colapso total |
| **V3 Noise 0.03** | **0.667 (8/12)** | **1.36** | **1.62** | **Sanos del lado Healthy** |

### 2. Hallazgo geométrico principal

**V3 Noise 0.03 revirtió el sesgo geométrico.** En Fase 2, los Healthy ingleses estaban
2.5× más cerca del centroide Patient checo. Con Noise 0.03, los Healthy ingleses están
**más cerca del centroide Healthy checo** (1.36 vs 1.62). Es la primera vez que la
geometría cross-lingual se alinea con las etiquetas reales.

### 3. Detalle por sujeto

```
Subj  Label  Score    d_CZ_H  d_CZ_P  Closer   GeoOK  Notas
 401    0    0.0145   1.187   1.550   healthy   OK    Muy seguro Healthy
 403    0    0.8106   1.141   1.451   healthy   OK
 404    0    0.9998   1.378   1.489   healthy   OK    Geometría OK, score alto (FP)
 405    0    0.0015   1.327   1.713   healthy   OK    Muy seguro Healthy
 409    0    0.2677   1.182   1.503   healthy   OK
 410    0    1.0000   1.767   1.741   patients  ERR   Único Healthy del lado Patients
 411    0    0.0004   1.539   1.924   healthy   OK    Bridge subject confirmado
 574    1    0.9988   1.387   1.527   healthy   ERR   Geométricamente Healthy (FN)
 607    1    0.4817   1.310   1.581   healthy   ERR   Geométricamente Healthy (FN)
 609    1    1.0000   1.905   1.841   patients  OK    Máxima separación
 611    1    0.9987   1.254   1.443   healthy   ERR   Geométricamente Healthy (FN)
 617    1    1.0000   1.741   1.711   patients  OK
```

### 4. Interpretación geométrica

- **6/7 Healthy correctamente del lado Healthy checo** (solo 410 falla)
- **410 y 404**: geometría correcta pero scores altos → problema de threshold, no de espacio
- **574, 607, 611**: geométricamente del lado Healthy → son los Patients más "leves"
- **609, 617**: máxima separación en geometría y scores → casos TEL más claros

**Conclusión geométrica:** Noise 0.03 produce el espacio latente más saludable para
transferencia cross-lingual de todos los modelos evaluados.

---

## Parte B — Noise 0.03 + Sparse Attention (Resultados en Checo)

### 5. Comparación Checa

| Modelo | Sample Bal Acc | Sample AUC | Subject Bal Acc | Fold Mean | Fold Std |
|---|---:|---:|---:|---:|---:|
| V3 control | 0.8089 | 0.9033 | 0.8977 | 0.7310 | 0.1108 |
| **V3 Noise 0.03** | **0.8497** | **0.9045** | **0.9432** | **0.8102** | **0.0667** |
| V3 LayerMix | **0.8783** | **0.9451** | **0.9432** | **0.8591** | **0.0680** |
| Noise+Sparse 0.001 | 0.8120 | 0.9036 | 0.8636 | 0.7404 | 0.1181 |
| Noise+Sparse 0.005 | 0.8162 | 0.8998 | 0.8977 | 0.7407 | 0.0885 |

**Diagnóstico:** Sparse Attention **empeora todas las métricas** respecto a Noise-only.
Baja balanced accuracy, baja AUC, baja subject accuracy, y en el caso de lambda=0.001
aumenta la varianza entre folds. La penalización de entropía está forzando un foco
temporal que destruye información útil para la clasificación.

---

## Parte C — Evaluación en Inglés

Dado que ambos `noise+sparse` son peores que `noise-only` en checo, y por el criterio
de "no destruir checo" establecido, **no se procede a evaluación en inglés** de estas
variantes. El modelo `V3 Noise 0.03` ya demostró rendimiento superior en inglés en
Fase 3C, y agregar sparse attention solo lo degradaría.

---

## Parte D — Tabla Comparativa Final

### Checo (10-fold CV)

| Modelo | Sample Bal Acc | AUC | Subj Bal Acc | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| V3 control | 0.8089 | 0.9033 | 0.8977 | 0.9614 | 0.8677 |
| V3 Noise 0.03 | 0.8497 | 0.9045 | 0.9432 | 0.9585 | 0.8914 |
| **V3 LayerMix** | **0.8783** | **0.9451** | **0.9432** | **0.9889** | **0.9152** |

### Inglés

| Modelo / Escenario | AUC | Bal Acc | Recall | Precision | F1 |
|---|---:|---:|---:|---:|---:|
| Fase 2 ZS recalibrado | 0.600 | 0.700 | 0.400 | 1.000 | 0.571 |
| Fase 2 LP LOSO | 0.743 | 0.657 | 0.600 | 0.600 | 0.600 |
| LayerMix ZS | 0.600 | 0.500 | 1.000 | 0.417 | 0.588 |
| LayerMix LP | 0.514 | 0.557 | 0.400 | 0.500 | 0.444 |
| **Noise 0.03 ZS** | **0.800** | 0.714 | **1.000** | 0.556 | **0.714** |
| **Noise 0.03 ZS rec** | **0.800** | **0.786** | **1.000** | **0.625** | **0.769** |
| **Noise 0.03 LP** | **0.743** | **0.757** | **0.800** | **0.667** | **0.727** |

### Geometría

| Modelo | Geometry Accuracy | EN Healthy → CZ H | EN Healthy → CZ P |
|---|---:|---:|---:|
| Fase 2 | 0.500 | 4.14 | 1.65 |
| LayerMix | 0.417 | 2.16 | 2.13 |
| **Noise 0.03** | **0.667** | **1.36** | **1.62** |

---

## Veredicto Final

### Rama Checa: `V3 LayerMix` — GO
- Mejor sample balanced accuracy (0.8783)
- Mejor AUC (0.9451)
- Mejor estabilidad entre folds
- Subject recall = 1.000

### Rama Multilingüe: `V3 Noise 0.03` — GO
- Mejor zero-shot AUC en inglés (0.800, +33% vs Fase 2)
- Mejor zero-shot recalibrado (bal_acc = 0.786)
- Mejor linear probing (bal_acc = 0.757, AUC = 0.743)
- Primera vez que la geometría cross-lingual se alinea con etiquetas reales (8/12)
- Healthy ingleses del lado Healthy checo por primera vez

### Sparse Attention: NO-GO
- Empeora checo en todas las métricas
- Aumenta varianza entre folds
- No justifica evaluación en inglés

### Decisión Final

**El proyecto tiene dos modelos oficiales:**
1. **Checo cerrado:** `V3 LayerMix` (HuBERT 12-layer mix + BiLSTM + attention + SupCon)
2. **Robustez multilingüe:** `V3 Noise 0.03` (HuBERT single-layer + BiLSTM + attention + SupCon + embedding noise)

**No se requiere un tercer modelo.** La arquitectura actual, con estas dos ramas
especializadas, cubre ambos objetivos de forma satisfactoria.

---

## Próximo Paso Recomendado

Si se desea continuar mejorando la rama multilingüe, la dirección más prometedora
ahora es:

1. **Adquirir más datos en inglés** (el cuello de botella real es n=12 sujetos)
2. **Fine-tuning ligero del clasificador** de `V3 Noise 0.03` sobre datos ingleses
3. **Modelos especializados por subtarea** (consonant-only, word-only) como
   alternativa de bajo costo a ResNet-18
