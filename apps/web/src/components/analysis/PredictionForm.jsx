import React from 'react'
import { FaUpload, FaPlay, FaTimes } from 'react-icons/fa'
import { statusLabels } from '../../constants/ui'

export function PredictionForm({ selectedFile, audioUrl, predictionState, errorMessage, onFileChange, onPredict, onClear }) {
  return (
    <div className="analysis-card analysis-upload" id="sample-upload">
      <div className="analysis-card__header">
        <span className="eyebrow">Entrada</span>
        <h3>Subir audio</h3>
      </div>

      <label className="upload-zone">
        <input type="file" accept=".wav,audio/wav" onChange={onFileChange} className="upload-zone__input" />
        <div className="upload-zone__content">
          <FaUpload size={28} className="upload-zone__icon" />
          <span className="upload-zone__text">
            {selectedFile ? selectedFile.name : 'Seleccionar archivo WAV'}
          </span>
        </div>
      </label>

      {audioUrl && (
        <audio controls src={audioUrl} className="audio-player" />
      )}

      <div className="analysis-actions">
        <button 
          className="button button--primary" 
          onClick={onPredict} 
          disabled={predictionState === 'processing' || !selectedFile}
        >
          <FaPlay /> Analizar
        </button>
        {selectedFile && (
          <button className="button button--ghost" onClick={onClear}>
            <FaTimes />
          </button>
        )}
      </div>

      <div className="analysis-status">
        <span className={`status-dot status-dot--${predictionState}`} />
        <span>{statusLabels[predictionState]}</span>
      </div>

      {errorMessage && (
        <div className="analysis-error">{errorMessage}</div>
      )}
    </div>
  )
}