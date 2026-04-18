import { useMemo, useState } from 'react'

import { API_BASE_URL } from '../constants/api'

export function usePrediction() {
  const [selectedFile, setSelectedFile] = useState(null)
  const [predictionState, setPredictionState] = useState('idle')
  const [prediction, setPrediction] = useState(null)
  const [errorMessage, setErrorMessage] = useState('')

  const probabilityEntries = useMemo(() => {
    if (!prediction?.probabilities) {
      return []
    }

    return Object.entries(prediction.probabilities).sort(([, a], [, b]) => Number(b) - Number(a))
  }, [prediction])

  function handleFileChange(event) {
    const file = event.target.files?.[0] || null

    setPrediction(null)
    setErrorMessage('')
    setPredictionState('idle')

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

  async function handlePredict() {
    if (!selectedFile) {
      setPredictionState('error')
      setErrorMessage('Selecciona un archivo WAV antes de ejecutar la inferencia.')
      return
    }

    setPredictionState('processing')
    setPrediction(null)
    setErrorMessage('')

    try {
      const formData = new FormData()
      formData.append('file', selectedFile)

      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        body: formData,
      })

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

  function clearState() {
    setSelectedFile(null)
    setPrediction(null)
    setPredictionState('idle')
    setErrorMessage('')
  }

  return {
    selectedFile,
    predictionState,
    prediction,
    errorMessage,
    probabilityEntries,
    handleFileChange,
    handlePredict,
    clearState,
  }
}
