import { useEffect, useState } from 'react'

import { API_BASE_URL } from '../constants/api'

const ACTIVE_MODEL_IDS = ['multilingual-cz-en', 'czech-layermix']

const OFFLINE_MODELS = ACTIVE_MODEL_IDS.map((id) => ({ id, name: id, languages: [] }))

export function useModels() {
  const [models, setModels] = useState(OFFLINE_MODELS)
  const [modelState, setModelState] = useState('loading')

  useEffect(() => {
    let cancelled = false

    async function loadModels() {
      try {
        const response = await fetch(`${API_BASE_URL}/v1/models`)
        const payload = await response.json()

        if (!response.ok) {
          throw new Error(payload.detail || 'No se pudo consultar el catálogo de modelos.')
        }

        const discoveredModels = payload.models
          .filter((model) => ACTIVE_MODEL_IDS.includes(model.id))
          .map((model) => ({
            ...model,
            name: model.name || model.id,
            languages: model.languages || [],
          }))

        if (discoveredModels.length === 0) {
          throw new Error('No hay modelos activos disponibles.')
        }

        if (!cancelled) {
          setModels(discoveredModels)
          setModelState('ready')
        }
      } catch {
        if (!cancelled) {
          setModels(OFFLINE_MODELS)
          setModelState('error')
        }
      }
    }

    loadModels()

    return () => {
      cancelled = true
    }
  }, [])

  return { models, modelState }
}
