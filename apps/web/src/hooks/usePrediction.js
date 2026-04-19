import { useEffect, useMemo, useState } from 'react'

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

export function usePrediction() {
  const [inputMode, setInputMode] = useState('upload')
  const [selectedFile, setSelectedFile] = useState(null)
  const [selectedSample, setSelectedSample] = useState(null)
  const [predictionState, setPredictionState] = useState('idle')
  const [prediction, setPrediction] = useState(null)
  const [errorMessage, setErrorMessage] = useState('')
  const [sampleAudios, setSampleAudios] = useState([])
  const [sampleFilters, setSampleFilters] = useState(INITIAL_SAMPLE_FILTERS)
  const [samplePagination, setSamplePagination] = useState(INITIAL_SAMPLE_PAGINATION)
  const [sampleState, setSampleState] = useState('idle')

  const probabilityEntries = useMemo(() => {
    if (!prediction?.probabilities) {
      return []
    }

    return Object.entries(prediction.probabilities).sort(([, a], [, b]) => Number(b) - Number(a))
  }, [prediction])

  const selectedSampleAudioUrl = selectedSample ? `${API_BASE_URL}/sample-audios/${selectedSample.id}/stream` : ''

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
  }

  function handleInputModeChange(nextMode) {
    setInputMode(nextMode)
    resetPredictionState()

    if (nextMode === 'upload') {
      setSelectedSample(null)
      return
    }

    setSelectedFile(null)
  }

  function handleFileChange(event) {
    const file = event.target.files?.[0] || null

    resetPredictionState()
    setInputMode('upload')
    setSelectedSample(null)

    if (!file) {
      setSelectedFile(null)
      return
    }

    if (!file.name.toLowerCase().endsWith('.wav')) {
      setSelectedFile(null)
      setPredictionState('error')
      setErrorMessage('Solo se aceptan archivos WAV.')
      return
    }

    setSelectedFile(file)
  }

  function handleSampleSelect(sample) {
    setInputMode('samples')
    setSelectedFile(null)
    setSelectedSample(sample)
    resetPredictionState()
  }

  async function handlePredict() {
    const isUploadMode = inputMode === 'upload'

    if (isUploadMode && !selectedFile) {
      setPredictionState('error')
      setErrorMessage('Selecciona un archivo WAV antes de ejecutar la inferencia.')
      return
    }

    if (!isUploadMode && !selectedSample) {
      setPredictionState('error')
      setErrorMessage('Selecciona una muestra LANNA antes de ejecutar la inferencia.')
      return
    }

    setPredictionState('processing')
    setPrediction(null)
    setErrorMessage('')

    try {
      let response

      if (isUploadMode) {
        const formData = new FormData()
        formData.append('file', selectedFile)

        response = await fetch(`${API_BASE_URL}/predict`, {
          method: 'POST',
          body: formData,
        })
      } else {
        response = await fetch(`${API_BASE_URL}/predict-sample/${selectedSample.id}`, {
          method: 'POST',
        })
      }

      const payload = await response.json()
      if (!response.ok) {
        throw new Error(payload.detail || 'No se pudo completar la inferencia.')
      }

      setPrediction(payload)
      setPredictionState('success')
    } catch (error) {
      setPredictionState('error')
      setErrorMessage(error.message || 'No se pudo conectar con la API.')
    }
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
    setSelectedFile(null)
    setSelectedSample(null)
    setPrediction(null)
    setPredictionState('idle')
    setErrorMessage('')
  }

  return {
    inputMode,
    selectedFile,
    selectedSample,
    selectedSampleAudioUrl,
    predictionState,
    prediction,
    errorMessage,
    probabilityEntries,
    sampleAudios,
    sampleFilters,
    samplePagination,
    sampleState,
    handleInputModeChange,
    handleFileChange,
    handlePredict,
    handleSampleSelect,
    handleSampleFilterChange,
    handleSamplePageChange,
    handleSamplePageSizeChange,
    clearState,
  }
}
