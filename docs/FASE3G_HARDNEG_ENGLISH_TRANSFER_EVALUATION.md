# FASE 3G: Evaluacion en Ingles de V3 LayerMix + hardneg_hnweight2x

**Objetivo:** Determinar si la mejora hard-negative (checo) es transferible al ingles
o puramente especifica del dominio checo.

**Checkpoint:** `v3_layermix_hardneg_hnweight2x_fold1_of_10.pt`
(fold 1, Czech sample bal_acc=0.9975, auc=0.9995)

**Fecha:** Mayo 2026

**Veredicto: NO-GO -- hardneg is Czech-specific.**

---

## 1. Resultados en Ingles

### Zero-shot

| Metrica | V3 LayerMix | V3 Noise 0.03 | Hardneg_hnweight2x |
|---|---:|---:|---:|
| Raw bal_acc (inherited thr) | 0.5000 | 0.7143 | 0.5000 |
| Recalibrado bal_acc | 0.5000 | **0.7857** | 0.5000 |
| AUC | 0.6000 | **0.8000** | 0.6571 |
| Recall | 1.0000 | 1.0000 | 1.0000 |
| Precision | 0.4167 | **0.5556** | 0.4167 |

**Diagnostico:** El hardneg zero-shot produce saturacion (todos los scores ~1.0), igual
que V3 LayerMix. AUC=0.657, ligeramente mejor que LayerMix (0.600) pero muy lejos de
Noise 0.03 (0.800). Ningun threshold puede separar Healthy de Patients.

### Linear Probe LOSO

| Metrica | V3 LayerMix | V3 Noise 0.03 | Hardneg_hnweight2x |
|---|---:|---:|---:|
| Balanced Accuracy | 0.5571 | **0.7571** | 0.4571 |
| AUC | 0.5143 | **0.7429** | 0.2000 |
| Recall | 0.4000 | **0.8000** | 0.2000 |
| Precision | 0.5000 | **0.6667** | 0.3333 |

**Diagnostico:** El linear probing del hardneg es el **peor resultado de todos los modelos**
evaluados en ingles. AUC=0.200 (menor que azar). El entrenamiento especializado en sanos
checos dificiles hace que los context vectors sean aun menos informativos para ingles.

### Geometria

| Metrica | V3 LayerMix | V3 Noise 0.03 | Hardneg_hnweight2x |
|---|---:|---:|---:|
| Geometry accuracy | 0.4167 | **0.6667** | 0.5833 |
| EN Healthy -> CZ H | 2.16 | 1.36 | 1.73 |
| EN Healthy -> CZ P | 2.13 | 1.62 | 1.81 |

**Diagnostico:** La geometria mejora ligeramente respecto a LayerMix (0.417 -> 0.583),
pero no alcanza a Noise 0.03 (0.667). Los sanos ingleses estan ligeramente mas cerca
del centroide Healthy checo, pero todos los scores siguen en ~1.0.

---

## 2. Per-Subject Detail

```
Subj  Label  Score    d_CZ_H  d_CZ_P  Closer   Geo  Notas
 401    0    0.9999   1.639   1.728   healthy   OK   Geometria OK, score saturado
 403    0    0.9999   1.707   1.793   healthy   OK
 404    0    1.0000   1.955   2.026   healthy   OK   Geometria OK, score saturado
 405    0    0.9998   1.629   1.718   healthy   OK
 409    0    0.9998   1.665   1.752   healthy   OK
 410    0    1.0000   1.918   1.990   healthy   OK   Geometria OK!
 411    0    0.9997   1.590   1.687   healthy   OK   Bridge
 574    1    1.0000   1.703   1.786   healthy   ERR  Geometricamente sano
 607    1    1.0000   1.632   1.721   healthy   ERR  Geometricamente sano
 609    1    1.0000   1.902   1.971   healthy   ERR  Geometricamente sano
 611    1    1.0000   1.863   1.938   healthy   ERR  Geometricamente sano
 617    1    1.0000   1.878   1.948   healthy   ERR  Geometricamente sano
```

---

## 3. Comparacion Final

| Modelo | ZS AUC | ZS Best Bal Acc | LP Bal Acc | LP AUC | Geometry |
|---|---:|---:|---:|---:|---:|
| Fase 2 baseline | 0.600 | 0.700 | 0.657 | 0.743 | 0.500 |
| V3 LayerMix | 0.600 | 0.500 | 0.557 | 0.514 | 0.417 |
| **V3 Noise 0.03** | **0.800** | **0.786** | **0.757** | **0.743** | **0.667** |
| V3 LM + hardneg | 0.657 | 0.500 | 0.457 | 0.200 | 0.583 |

---

## 4. Conclusion

**La mejora hard-negative NO es transferible al ingles.**

El entrenamiento especializado en sanos checos dificiles (H26-H29) corrigio errores
especificos del dominio checo, pero al hacerlo, hizo que los context vectors del modelo
fueran aun menos informativos para ingles. El linear probe AUC bajo de 0.514 (LayerMix)
a 0.200 (hardneg), el peor resultado registrado.

### Veredicto final

**NO-GO -- hardneg is Czech-specific.**

La mejora hard-negative debe mantenerse exclusivamente para la rama checa, donde es
altamente efectiva (subject bal_acc 0.9773). Para transferencia multilingue, V3 Noise
0.03 sigue siendo la rama optima.

---

## 5. Tabla Final del Proyecto

| Objetivo | Modelo final | Metrica clave |
|---|---|---|
| Checo intra-dominio | **V3 LayerMix + hardneg_hnweight2x** | subject bal_acc 0.9773 |
| Robustez multilingue / ingles | **V3 Noise 0.03** | zero-shot AUC 0.800 |
| Sparse Attention | NO-GO | — |
| Sub-task specialization | NO-GO | — |
| ResNet-18 Kotarba benchmark | Benchmark util, no rama oficial | — |
| Hardneg + transferencia a ingles | **NO-GO (checo-especifico)** | LP AUC 0.200 |
