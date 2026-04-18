import { useEffect, useState } from 'react'

export function useAudioPreview(file) {
  const [audioUrl, setAudioUrl] = useState('')

  useEffect(() => {
    if (!file) {
      setAudioUrl('')
      return undefined
    }

    const nextUrl = URL.createObjectURL(file)
    setAudioUrl(nextUrl)

    return () => URL.revokeObjectURL(nextUrl)
  }, [file])

  return audioUrl
}
