import React, { useEffect, useRef } from 'react'
import WaveSurfer from 'wavesurfer.js'
import { FaPlay, FaPause } from 'react-icons/fa'

export function AudioVisualizer({ audioUrl }) {
  const waveformRef = useRef(null)
  const wavesurferRef = useRef(null)
  const [isPlaying, setIsPlaying] = React.useState(false)

  useEffect(() => {
    if (!audioUrl || !waveformRef.current) return

    wavesurferRef.current = WaveSurfer.create({
      container: waveformRef.current,
      waveColor: '#60a5fa',
      progressColor: '#2563eb',
      cursorColor: '#1d4ed8',
      barWidth: 2,
      barGap: 1,
      barRadius: 2,
      height: 80,
      normalize: true,
    })

    wavesurferRef.current.load(audioUrl)

    wavesurferRef.current.on('play', () => setIsPlaying(true))
    wavesurferRef.current.on('pause', () => setIsPlaying(false))
    wavesurferRef.current.on('finish', () => setIsPlaying(false))

    return () => {
      wavesurferRef.current?.destroy()
    }
  }, [audioUrl])

  const togglePlayback = () => {
    wavesurferRef.current?.playPause()
  }

  if (!audioUrl) return null

  return (
    <div className="analysis-card audio-visualizer">
      <div className="analysis-card__header">
        <span className="eyebrow">Visualización</span>
        <h3>Forma de onda</h3>
      </div>
      <div className="waveform-container" ref={waveformRef} />
      <button className="button button--ghost waveform-controls" onClick={togglePlayback}>
        {isPlaying ? <FaPause size={16} /> : <FaPlay size={16} />}
        <span>{isPlaying ? 'Pausar' : 'Reproducir'}</span>
      </button>
    </div>
  )
}