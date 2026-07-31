import { useEffect, useState } from 'react'

import { API_BASE_URL } from '../constants/api'

export function useDemoCapability() {
  const [demoLibraryEnabled, setDemoLibraryEnabled] = useState(false)

  useEffect(() => {
    let cancelled = false

    async function loadCapability() {
      try {
        const response = await fetch(`${API_BASE_URL}/v1/capabilities`)
        if (!response.ok) return

        const payload = await response.json()
        if (!cancelled) setDemoLibraryEnabled(payload.demo_library_enabled === true)
      } catch {
        // The upload-only interface remains available when capabilities cannot be read.
      }
    }

    loadCapability()
    return () => {
      cancelled = true
    }
  }, [])

  return demoLibraryEnabled
}
