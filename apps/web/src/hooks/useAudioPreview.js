import { useEffect, useState } from 'react'

export function useAudioPreview(files) {
  const [audioUrl, setAudioUrl] = useState('')

  useEffect(() => {
    const file = files?.[0]

    if (!file) {
      setAudioUrl('')
      return undefined
    }

    const nextUrl = URL.createObjectURL(file)
    setAudioUrl(nextUrl)

    return () => URL.revokeObjectURL(nextUrl)
  }, [files])

  return audioUrl
}
