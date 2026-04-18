export const pipelineSteps = [
  ['Ingreso de muestra', 'Recepción y validación de un archivo WAV para el análisis de indicadores vocales asociados al TEL.'],
  ['Preprocesamiento', 'Remuestreo, filtrado, normalización y ajuste temporal para obtener una señal consistente.'],
  ['Embeddings acústicos', 'Extracción de representaciones con HuBERT para capturar patrones relevantes de la producción vocal.'],
  ['Identificación TEL', 'Escalado y clasificación final con ONNX para estimar la clase y su nivel de confianza.'],
]

export const statusLabels = {
  idle: 'Listo',
  processing: 'Procesando',
  success: 'Completado',
  error: 'Error',
}

export const summaryItems = [
  ['Entrada', 'Audio WAV'],
  ['Salida', 'Clase y confianza'],
  ['Motor', 'HuBERT + ONNX'],
  ['Modo', 'Dashboard clínico'],
]

export const telHighlights = [
  ['Objetivo', 'Visualizar resultados clave para identificación de TEL.'],
  ['Prioridad', 'Gráficos, métricas y lectura rápida del caso.'],
  ['Uso', 'Apoyo académico y demostración técnica.'],
]

export const heroNotes = [
  ['Muestra', 'Archivo WAV individual'],
  ['Resultado', 'Clase, confianza y tiempos'],
  ['Lectura', 'Dashboard resumido del caso'],
]

export const evidenceSignals = [
  'consistencia temporal de la señal',
  'patrones acústicos extraídos por embeddings',
  'probabilidades por clase para lectura comparativa',
]

export const inputRequirements = [
  ['Formato esperado', 'El sistema recibe una muestra individual en formato WAV para mantener compatibilidad con el flujo de procesamiento.'],
  ['Condición de la grabación', 'Se recomienda una señal estable, con poco ruido y sin superposición de voces para evitar lecturas distorsionadas.'],
  ['Resultado de la carga', 'Una vez cargado el archivo, el workspace prepara la muestra para ejecutar la inferencia y mostrar los resultados.'],
]

export const outputInsights = [
  ['Clase estimada', 'Muestra la categoría que el modelo considera más probable para la muestra procesada.'],
  ['Confianza', 'Indica qué tan sólida es la predicción dentro del conjunto de probabilidades calculadas.'],
  ['Tiempos del proceso', 'Permiten revisar cuánto tarda cada etapa del pipeline, desde el preprocesamiento hasta la clasificación.'],
]
