import { useEffect, useState } from 'react'

import { API_BASE_URL } from '../constants/api'

export function useHealthCheck() {
  const [health, setHealth] = useState({ status: 'loading', modelVersion: null, message: 'Verificando API' })

  useEffect(() => {
    let cancelled = false

    async function checkHealth() {
      try {
        const response = await fetch(`${API_BASE_URL}/health`)
        if (!response.ok) {
          throw new Error('No disponible')
        }

        const payload = await response.json()
        if (!cancelled) {
          setHealth({ status: 'ready', modelVersion: payload.model_version, message: 'API disponible' })
        }
      } catch {
        if (!cancelled) {
          setHealth({ status: 'error', modelVersion: null, message: 'API no disponible' })
        }
      }
    }

    checkHealth()

    return () => {
      cancelled = true
    }
  }, [])

  return health
}
