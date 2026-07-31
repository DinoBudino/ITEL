import { useEffect, useMemo, useRef, useState } from 'react'

import { API_BASE_URL } from '../constants/api'

const INITIAL_PAGINATION = { page: 1, pageSize: 20, total: 0, totalPages: 0 }
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

export function useDemoLibrary(models, modelState, active) {
  const [selectedModelId, setSelectedModelId] = useState('')
  const [selectedLanguage, setSelectedLanguage] = useState('')
  const [selectedReferenceLabel, setSelectedReferenceLabel] = useState('')
  const [subjects, setSubjects] = useState([])
  const [selectedSubject, setSelectedSubject] = useState(null)
  const [pagination, setPagination] = useState(INITIAL_PAGINATION)
  const [libraryState, setLibraryState] = useState('idle')
  const [predictionState, setPredictionState] = useState('idle')
  const [prediction, setPrediction] = useState(null)
  const [streamProgress, setStreamProgress] = useState(null)
  const [errorMessage, setErrorMessage] = useState('')
  const predictionControllerRef = useRef(null)

  const selectedModel = models.find((model) => model.id === selectedModelId)
  const availableLanguages = (selectedModel?.languages || []).filter((language) => ['cs', 'en'].includes(language))
  const modelAvailable = selectedModel?.available === true
  const probabilityEntries = useMemo(() => {
    if (!prediction?.probabilities) return []
    return Object.entries(prediction.probabilities).sort(([, a], [, b]) => Number(b) - Number(a))
  }, [prediction])

  useEffect(() => () => predictionControllerRef.current?.abort(), [])

  useEffect(() => {
    if (models.length > 0 && !models.some((model) => model.id === selectedModelId)) {
      setSelectedModelId(models[0].id)
    }
  }, [models, selectedModelId])

  useEffect(() => {
    if (selectedSubject) return
    if (!availableLanguages.includes(selectedLanguage)) {
      setSelectedLanguage(availableLanguages[0] || '')
    }
  }, [availableLanguages, selectedLanguage, selectedSubject])

  useEffect(() => {
    if (!active || modelState !== 'ready' || !selectedModelId || !selectedLanguage) return undefined

    let cancelled = false
    async function loadSubjects() {
      setLibraryState('loading')
      setErrorMessage('')
      try {
        const params = new URLSearchParams({
          model_id: selectedModelId,
          language: selectedLanguage,
          page: String(pagination.page),
          page_size: String(pagination.pageSize),
        })
        if (selectedReferenceLabel) params.set('reference_label', selectedReferenceLabel)
        const response = await fetch(`${API_BASE_URL}/v1/demo/subjects?${params}`)
        const payload = await response.json().catch(() => ({}))
        if (!response.ok) throw new Error(payload.detail || 'No se pudo cargar la base de pruebas.')
        if (cancelled) return

        setSubjects(payload.items || [])
        setPagination((current) => ({
          ...current,
          page: payload.page,
          pageSize: payload.page_size,
          total: payload.total,
          totalPages: payload.total_pages,
        }))
        setLibraryState('ready')
      } catch (error) {
        if (cancelled) return
        setSubjects([])
        setLibraryState('error')
        setErrorMessage(error.message || 'No se pudo cargar la base de pruebas.')
      }
    }

    loadSubjects()
    return () => {
      cancelled = true
    }
  }, [active, modelState, pagination.page, pagination.pageSize, selectedLanguage, selectedModelId, selectedReferenceLabel])

  function resetPrediction() {
    setPrediction(null)
    setPredictionState('idle')
    setStreamProgress(null)
    setErrorMessage('')
  }

  function handleModelChange(modelId) {
    setSelectedSubject(null)
    setSelectedModelId(modelId)
    setPagination(INITIAL_PAGINATION)
    resetPrediction()
  }

  function handleLanguageChange(language) {
    if (selectedSubject) return
    setSelectedLanguage(language)
    setPagination(INITIAL_PAGINATION)
    resetPrediction()
  }

  function handleReferenceLabelChange(referenceLabel) {
    setSelectedSubject(null)
    setSelectedReferenceLabel(referenceLabel)
    setPagination(INITIAL_PAGINATION)
    resetPrediction()
  }

  async function handleSubjectSelect(subject) {
    resetPrediction()
    setSelectedSubject(subject)
    setSelectedLanguage(subject.language)
    setLibraryState('loading-detail')
    try {
      const response = await fetch(`${API_BASE_URL}/v1/demo/subjects/${encodeURIComponent(subject.token)}`)
      const payload = await response.json().catch(() => ({}))
      if (!response.ok) throw new Error(payload.detail || 'No se pudo cargar el sujeto de prueba.')
      setSelectedSubject(payload)
      setSelectedLanguage(payload.language)
      setLibraryState('ready')
    } catch (error) {
      setSelectedSubject(null)
      setLibraryState('error')
      setErrorMessage(error.message || 'No se pudo cargar el sujeto de prueba.')
    }
  }

  function handleClearSubject() {
    setSelectedSubject(null)
    resetPrediction()
  }

  async function handlePredict() {
    if (!selectedSubject || modelState !== 'ready' || !selectedModelId || !modelAvailable) return

    setPredictionState('processing')
    setPrediction(null)
    setErrorMessage('')
    setStreamProgress({ ...INITIAL_PROGRESS, startedAt: Date.now() })
    const controller = new AbortController()
    predictionControllerRef.current = controller

    try {
      const formData = new FormData()
      formData.append('model_id', selectedModelId)
      const response = await fetch(`${API_BASE_URL}/v1/demo/subjects/${encodeURIComponent(selectedSubject.token)}/predict/stream`, {
        method: 'POST',
        body: formData,
        signal: controller.signal,
      })
      if (!response.ok) {
        const payload = await response.json().catch(() => null)
        throw new Error(payload?.detail || 'No se pudo completar la inferencia.')
      }
      if (!response.body) throw new Error('El navegador no pudo leer la respuesta en streaming.')

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let receivedResult = false

      function parseLine(line) {
        if (!line.trim()) return
        let event
        try {
          event = JSON.parse(line)
        } catch {
          throw new Error('La API devolvió un evento de progreso inválido.')
        }
        if (event.type === 'progress') {
          setStreamProgress((current) => ({ ...current, ...event }))
        } else if (event.type === 'result') {
          receivedResult = true
          setStreamProgress((current) => ({ ...current, ...event, progress: 100, indeterminate: false }))
          setPrediction(event.data)
          setPredictionState('success')
        } else if (event.type === 'error') {
          throw new Error(event.detail || 'No se pudo completar la inferencia.')
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
      if (!receivedResult) throw new Error('La conexión terminó antes de recibir el resultado.')
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
      if (predictionControllerRef.current === controller) predictionControllerRef.current = null
    }
  }

  return {
    selectedModelId,
    selectedLanguage,
    selectedReferenceLabel,
    availableLanguages,
    modelAvailable,
    subjects,
    selectedSubject,
    pagination,
    libraryState,
    predictionState,
    prediction,
    streamProgress,
    errorMessage,
    probabilityEntries,
    handleModelChange,
    handleLanguageChange,
    handleReferenceLabelChange,
    handleSubjectSelect,
    handleClearSubject,
    handlePageChange: (page) => setPagination((current) => ({ ...current, page })),
    handlePredict,
    handleCancel: () => predictionControllerRef.current?.abort(),
  }
}
