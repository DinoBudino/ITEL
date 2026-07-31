import { useEffect, useMemo, useRef, useState } from 'react'

import { API_BASE_URL } from '../constants/api'

const INITIAL_SAMPLE_FILTERS = {
  classLabel: 'All',
  query: '',
}

const INITIAL_SAMPLE_PAGINATION = {
  page: 1,
  pageSize: 5,
  total: 0,
  totalPages: 0,
}

const INITIAL_PROGRESS = {
  stage: 'starting',
  progress: null,
  indeterminate: true,
  message: 'Iniciando inferencia...',
  file_index: null,
  file_count: null,
  elapsed_ms: null,
  cache_hits: null,
  cache_misses: null,
}

export function usePrediction(models, modelState) {
  const [inputMode, setInputMode] = useState('upload')
  const [selectedFiles, setSelectedFiles] = useState([])
  const [selectedSample, setSelectedSample] = useState(null)
  const [selectedModelId, setSelectedModelId] = useState('')
  const [selectedLanguage, setSelectedLanguage] = useState('')
  const [predictionState, setPredictionState] = useState('idle')
  const [prediction, setPrediction] = useState(null)
  const [errorMessage, setErrorMessage] = useState('')
  const [streamProgress, setStreamProgress] = useState(null)
  const [sampleAudios, setSampleAudios] = useState([])
  const [sampleFilters, setSampleFilters] = useState(INITIAL_SAMPLE_FILTERS)
  const [samplePagination, setSamplePagination] = useState(INITIAL_SAMPLE_PAGINATION)
  const [sampleState, setSampleState] = useState('idle')
  const predictionControllerRef = useRef(null)

  useEffect(() => () => predictionControllerRef.current?.abort(), [])

  const probabilityEntries = useMemo(() => {
    if (!prediction?.probabilities) {
      return []
    }

    return Object.entries(prediction.probabilities).sort(([, a], [, b]) => Number(b) - Number(a))
  }, [prediction])

  const selectedSampleAudioUrl = selectedSample ? `${API_BASE_URL}/sample-audios/${selectedSample.id}/stream` : ''

  const selectedModel = models.find((model) => model.id === selectedModelId)
  const availableLanguages = selectedModel?.languages || []
  const modelAvailable = selectedModel?.available === true

  useEffect(() => {
    if (models.length === 0) return

    if (!models.some((model) => model.id === selectedModelId)) {
      setSelectedModelId(models[0].id)
    }
  }, [models, selectedModelId])

  useEffect(() => {
    if (availableLanguages.length === 0) {
      setSelectedLanguage('')
      return
    }

    if (!availableLanguages.includes(selectedLanguage)) {
      setSelectedLanguage(availableLanguages[0])
    }
  }, [availableLanguages, selectedLanguage])

  useEffect(() => {
    if (inputMode !== 'samples') {
      return undefined
    }

    let isCancelled = false

    async function loadSampleAudios() {
      setSampleState('loading')

      try {
        const params = new URLSearchParams({
          page: String(samplePagination.page),
          page_size: String(samplePagination.pageSize),
        })

        if (sampleFilters.classLabel !== 'All') {
          params.set('class_label', sampleFilters.classLabel)
        }

        if (sampleFilters.query.trim()) {
          params.set('query', sampleFilters.query.trim())
        }

        const response = await fetch(`${API_BASE_URL}/sample-audios?${params.toString()}`)
        const payload = await response.json()

        if (!response.ok) {
          throw new Error(payload.detail || 'No se pudieron cargar las muestras LANNA.')
        }

        if (isCancelled) {
          return
        }

        setSampleAudios(payload.items || [])
        setSamplePagination((current) => ({
          ...current,
          page: payload.page,
          pageSize: payload.page_size,
          total: payload.total,
          totalPages: payload.total_pages,
        }))
        setSampleState('ready')
      } catch (error) {
        if (isCancelled) {
          return
        }

        setSampleAudios([])
        setSampleState('error')
        setErrorMessage(error.message || 'No se pudieron cargar las muestras LANNA.')
      }
    }

    loadSampleAudios()

    return () => {
      isCancelled = true
    }
  }, [inputMode, sampleFilters.classLabel, sampleFilters.query, samplePagination.page, samplePagination.pageSize])

  function resetPredictionState() {
    setPrediction(null)
    setErrorMessage('')
    setPredictionState('idle')
    setStreamProgress(null)
  }

  function handleInputModeChange(nextMode) {
    setInputMode(nextMode)
    resetPredictionState()

    if (nextMode === 'upload') {
      setSelectedSample(null)
      return
    }

    setSelectedFiles([])
  }

  function handleFileChange(event) {
    const files = Array.from(event.target.files || [])

    resetPredictionState()
    setInputMode('upload')
    setSelectedSample(null)

    if (files.length === 0) {
      setSelectedFiles([])
      return
    }

    if (files.some((file) => !file.name.toLowerCase().endsWith('.wav'))) {
      setSelectedFiles([])
      setPredictionState('error')
      setErrorMessage('Todos los archivos deben estar en formato WAV.')
      return
    }

    setSelectedFiles(files)
  }

  function handleSampleSelect(sample) {
    setInputMode('samples')
    setSelectedFiles([])
    setSelectedSample(sample)
    resetPredictionState()
  }

  function handleModelChange(modelId) {
    setSelectedModelId(modelId)
    resetPredictionState()
  }

  function handleLanguageChange(language) {
    setSelectedLanguage(language)
    resetPredictionState()
  }

  async function handlePredict() {
    const isUploadMode = inputMode === 'upload'

    if (isUploadMode && selectedFiles.length === 0) {
      setPredictionState('error')
      setErrorMessage('Selecciona uno o más archivos WAV antes de ejecutar la inferencia.')
      return
    }

    if (!isUploadMode && !selectedSample) {
      setPredictionState('error')
      setErrorMessage('Selecciona una muestra LANNA antes de ejecutar la inferencia.')
      return
    }

    if (modelState !== 'ready' || !selectedModelId || !selectedLanguage) {
      setPredictionState('error')
      setErrorMessage('El catálogo de modelos no está disponible. Verifica la conexión con la API.')
      return
    }

    if (!modelAvailable) {
      setPredictionState('error')
      setErrorMessage('El bundle de este modelo todavía no está disponible o no superó su validación.')
      return
    }

    setPredictionState('processing')
    setPrediction(null)
    setErrorMessage('')
    setStreamProgress({ ...INITIAL_PROGRESS, startedAt: Date.now() })

    const controller = new AbortController()
    predictionControllerRef.current = controller

    try {
      const formData = new FormData()
      formData.append('model_id', selectedModelId)
      formData.append('language', selectedLanguage)

      if (isUploadMode) {
        selectedFiles.forEach((file) => formData.append('files', file))
      } else {
        const sampleResponse = await fetch(selectedSampleAudioUrl, { signal: controller.signal })
        if (!sampleResponse.ok) {
          throw new Error('No se pudo cargar la muestra LANNA seleccionada.')
        }
        const sampleBlob = await sampleResponse.blob()
        formData.append('files', sampleBlob, selectedSample.filename)
      }

      const response = await fetch(`${API_BASE_URL}/v1/predictions/subject/stream`, {
        method: 'POST',
        body: formData,
        signal: controller.signal,
      })

      if (!response.ok) {
        const payload = await response.json().catch(() => null)
        throw new Error(payload?.detail || 'No se pudo completar la inferencia.')
      }

      if (!response.body) {
        throw new Error('El navegador no pudo leer la respuesta en streaming.')
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let receivedResult = false

      function handleStreamEvent(event) {
        if (event.type === 'progress') {
          setStreamProgress((current) => ({ ...current, ...event }))
          return
        }

        if (event.type === 'result') {
          receivedResult = true
          setStreamProgress((current) => ({ ...current, ...event, progress: 100, indeterminate: false }))
          setPrediction(event.data)
          setPredictionState('success')
          return
        }

        if (event.type === 'error') {
          throw new Error(event.detail || 'No se pudo completar la inferencia.')
        }
      }

      function parseLine(line) {
        if (!line.trim()) return

        try {
          handleStreamEvent(JSON.parse(line))
        } catch (error) {
          if (error instanceof SyntaxError) {
            throw new Error('La API devolvió un evento de progreso inválido.')
          }
          throw error
        }
      }

      while (true) {
        const { value, done } = await reader.read()
        buffer += decoder.decode(value || new Uint8Array(), { stream: !done })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''
        lines.forEach(parseLine)

        if (done) break
      }

      parseLine(buffer)
      if (!receivedResult) {
        throw new Error('La conexión terminó antes de recibir el resultado.')
      }
    } catch (error) {
      if (error.name === 'AbortError') {
        setPredictionState('cancelled')
        setErrorMessage('Inferencia cancelada.')
      } else {
        controller.abort()
        setPredictionState('error')
        setErrorMessage(error.message || 'No se pudo conectar con la API.')
      }
    } finally {
      if (predictionControllerRef.current === controller) {
        predictionControllerRef.current = null
      }
    }
  }

  function handleCancel() {
    predictionControllerRef.current?.abort()
  }

  function handleSampleFilterChange(key, value) {
    setSampleFilters((current) => ({
      ...current,
      [key]: value,
    }))
    setSamplePagination((current) => ({
      ...current,
      page: 1,
    }))
  }

  function handleSamplePageChange(nextPage) {
    setSamplePagination((current) => ({
      ...current,
      page: nextPage,
    }))
  }

  function handleSamplePageSizeChange(nextPageSize) {
    setSamplePagination((current) => ({
      ...current,
      page: 1,
      pageSize: nextPageSize,
    }))
  }

  function clearState() {
    setSelectedFiles([])
    setSelectedSample(null)
    setPrediction(null)
    setPredictionState('idle')
    setErrorMessage('')
    setStreamProgress(null)
  }

  return {
    inputMode,
    selectedFiles,
    selectedSample,
    selectedModelId,
    selectedLanguage,
    availableLanguages,
    modelAvailable,
    selectedSampleAudioUrl,
    predictionState,
    prediction,
    streamProgress,
    errorMessage,
    probabilityEntries,
    sampleAudios,
    sampleFilters,
    samplePagination,
    sampleState,
    handleInputModeChange,
    handleFileChange,
    handleModelChange,
    handleLanguageChange,
    handlePredict,
    handleCancel,
    handleSampleSelect,
    handleSampleFilterChange,
    handleSamplePageChange,
    handleSamplePageSizeChange,
    clearState,
  }
}
