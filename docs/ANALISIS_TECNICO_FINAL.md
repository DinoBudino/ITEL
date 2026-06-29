# ANALISIS TECNICO FINAL: Proyecto ITEL

**Fases 1-3 completadas. Prueba con ingles post-LayerMix pendiente.**

---

## FASE 1: Dataset LANNA -- Checo

**Modelo ganador:** HuBERT-base (congelado) + Bi-LSTM (2 capas, hidden=256) + Single-Head Attention + BCE + SupCon (lambda=0.1)

**Protocolo:** StratifiedGroupKFold, 5-fold, speaker-independent

| Metrica | Sample Level | Subject Level |
|---|---|---|
| Accuracy | 0.8603 | 0.8947 |
| Balanced Accuracy | 0.8577 | 0.8944 |
| AUC | 0.9216 | 0.9556 |
| F1 | 0.8752 | 0.9000 |
| Recall | 0.8820 | 0.9000 |

**Artefacto:** `research/checkpoints/best_model_checo_fase1.pt`

---

## FASE 2: Validacion Multilingue -- Transferencia a Ingles

**Dataset:** 12 ninos (7 Healthy, 5 Patients), picture-description, audio natural-length

**Enfoque escalonado:** Zero-shot -> Linear Probing (LOSO) -> anaalisis geometrico

| Metodo | Balanced Accuracy | AUC | Interpretacion |
|---|---|---|---|
| Zero-shot (thr=0.20) | 0.500 | 0.600 | Colapso: todo Patients |
| Zero-shot (thr=0.95 recalibrado) | 0.700 | 0.600 | Senial desplazada, no perdida |
| Linear Probe LOSO (LR) | 0.657 | **0.743** | AUC +24%: senial transferible |

**Hallazgos geometricos:**
- El sujeto 411 es el unico "puente": Healthy ingles mas cercano al centroide Healthy checo
- 11/12 sujetos caen mas cerca del centroide Patient checo (domain shift fonetico)
- El linear probing rescato 2 pacientes (574, 611) que el zero-shot no detectaba

**Conclusion Fase 2:** Los marcadores acusticos del backbone HuBERT+Bi-LSTM contienen
informacion diagnostica transferible entre idiomas, pero la frontera de decision
requiere mas datos locales.

**Documento completo:** `FASE2_MULTILINGUAL_FINDINGS.md`

---

## FASE 3: Mejoras Arquitectonicas

### Mejora A (LayerMix) -- VALIDADA

**Que hace:** Suma pesada aprendible de las 12 capas de HuBERT (ScalarMix) en vez de usar solo la ultima.

**Resultado limpio (10-fold, 12 capas reales):**

| Metrica | V3 Control | V3 LayerMix | Delta |
|---|---:|---:|---:|
| Sample Accuracy | 0.8316 | **0.8948** | **+0.0632** |
| Sample Balanced Accuracy | 0.8089 | **0.8783** | **+0.0694** |
| Sample AUC | 0.9033 | **0.9451** | **+0.0418** |
| Sample F1 | 0.8677 | **0.9152** | **+0.0475** |
| Subject Accuracy | 0.9082 | **0.9490** | **+0.0408** |
| Subject AUC | 0.9621 | **0.9739** | **+0.0118** |
| Mean Fold Bal Acc | 0.7310 +/- 0.1108 | **0.8591 +/- 0.0680** | +12.8pp, -39% std |

**Veredicto:** LayerMix es una mejora arquitectonica significativa. Mejora todas las
metricas principales y reduce la varianza entre folds.

### Mejora B (Embedding Noise) -- PENDIENTE

**Que hace:** Ruido Gaussiano en embeddings durante entrenamiento.

**Estado:** Compila y entrena. No se ha ejecutado experimento completo 10-fold.

### Mejora C (Sparse Attention) -- PENDIENTE

**Que hace:** Penalizacion de entropia en pesos de atencion.

**Estado:** Compila y entrena. No se ha ejecutado experimento completo 10-fold.

**Documento completo:** `FASE3_ARCHITECTURE_IMPROVEMENTS.md`

---

## Proximos Pasos (en orden de prioridad)

1. **Prueba con ingles post-LayerMix:** Evaluar el checkpoint V3 LayerMix sobre el dataset ingles (zero-shot + linear probing) y comparar contra los resultados de Fase 2.

2. **Experimento Mejora B (Noise):** `python -m research.scripts.train_optimized_v3 --embedding-noise-std 0.05 --epochs 30 --n-splits 10`

3. **Experimento Mejora C (Sparse):** `python -m research.scripts.train_optimized_v3 --attention-entropy-lambda 0.01 --epochs 30 --n-splits 10`

4. **Mejora A+B+C combinadas:** `python -m research.scripts.train_optimized_v3 --use-layer-mix --embedding-noise-std 0.05 --attention-entropy-lambda 0.01 --sequence-manifest research/processed_data/manifests/hubert_sequences_all_layers.csv --epochs 30 --n-splits 10`

5. **Replica ResNet-18 (Kotarba):** Benchmark paralelo con espectrogramas log-mel + CNN para comparar contra el 99.48% reportado.

---

## Artefactos Clave

| Archivo | Contenido |
|---|---|
| `research/checkpoints/best_model_checo_fase1.pt` | Modelo Fase 1 |
| `research/checkpoints/v3_final/layermix/` | 10 checkpoints V3 LayerMix |
| `research/scripts/train_optimized_v3.py` | Script V3 con 3 mejoras toggleables |
| `FASE2_MULTILINGUAL_FINDINGS.md` | Reporte Fase 2 |
| `FASE3_ARCHITECTURE_IMPROVEMENTS.md` | Reporte Fase 3 |
