import React from 'react'
import { FaCheckCircle, FaClock, FaChartBar, FaMicrochip, FaUpload, FaPlay, FaFileAudio, FaWaveSquare, FaSignal, FaBrain, FaDatabase } from 'react-icons/fa'
import { formatMs, formatPercent } from '../../utils/formatters'

function getConfidenceLevel(confidence) {
  const numericConfidence = Number(confidence || 0)
  if (numericConfidence >= 0.85) return 'Alta'
  if (numericConfidence >= 0.65) return 'Media'
  return 'Exploratoria'
}

function EmptyResults() {
  return (
    <div className="result-empty">
      <div className="result-empty__steps">
        <div className="result-step">
          <div className="result-step__icon">
            <FaUpload size={20} />
          </div>
          <div className="result-step__content">
            <strong>1. Sube el audio</strong>
            <p>Selecciona un archivo WAV de voz</p>
          </div>
        </div>
        <div className="result-step">
          <div className="result-step__icon">
            <FaPlay size={20} />
          </div>
          <div className="result-step__content">
            <strong>2. Ejecuta el análisis</strong>
            <p>Procesa la muestra de voz</p>
          </div>
        </div>
        <div className="result-step">
          <div className="result-step__icon">
            <FaFileAudio size={20} />
          </div>
          <div className="result-step__content">
            <strong>3. Obtén resultados</strong>
            <p>Recibe la predicción con confianza</p>
          </div>
        </div>
      </div>
    </div>
  )
}

export function ResultsPanel({ prediction, probabilityEntries }) {
  const confidenceLevel = getConfidenceLevel(prediction?.confidence)

  return (
    <div className="analysis-card analysis-results" id="results">
      <div className="analysis-card__header">
        <span className="eyebrow">Resultados</span>
        <h3>Dashboard del modelo</h3>
      </div>

      {prediction ? (
        <>
          <div className="result-main">
            <div className="result-main__label">
              <FaCheckCircle size={20} />
              <span>Clase estimada</span>
            </div>
            <strong className="result-main__value">{prediction.predicted_label}</strong>
            <p className="result-main__description">
              Confianza: {formatPercent(prediction.confidence)} • Nivel: {confidenceLevel}
            </p>
          </div>

          <div className="result-metrics">
            <div className="result-metrics__item">
              <FaClock size={18} />
              <span className="result-metrics__label">Tiempo total</span>
              <strong className="result-metrics__value">{formatMs(prediction.timings_ms?.total)}</strong>
            </div>
            <div className="result-metrics__item">
              <FaWaveSquare size={18} />
              <span className="result-metrics__label">Ventana análisis</span>
              <strong className="result-metrics__value">{prediction.timings_ms?.embedding ? '768ms' : '-'}</strong>
            </div>
            <div className="result-metrics__item">
              <FaSignal size={18} />
              <span className="result-metrics__label">Threshold</span>
              <strong className="result-metrics__value">{prediction.confidence >= 0.7 ? '≥70%' : '<'+'70%'}</strong>
            </div>
          </div>

          <div className="result-metrics">
            <div className="result-metrics__item">
              <FaBrain size={18} />
              <span className="result-metrics__label">Modelo</span>
              <strong className="result-metrics__value">{prediction.metadata?.model_version || 'itelv5'}</strong>
            </div>
            <div className="result-metrics__item">
              <FaDatabase size={18} />
              <span className="result-metrics__label">Audio size</span>
              <strong className="result-metrics__value">{prediction.metadata?.duration_seconds ? `${(prediction.metadata.duration_seconds * 16).toFixed(0)} KB` : '-'}</strong>
            </div>
            <div className="result-metrics__item">
              <FaMicrochip size={18} />
              <span className="result-metrics__label">Veces embedding</span>
              <strong className="result-metrics__value">{prediction.timings_ms?.embedding ? Math.ceil(Number(prediction.metadata?.duration_seconds || 0) / 0.768) : '-'}</strong>
            </div>
          </div>

          <div className="result-distribution">
            <h4>Distribución de probabilidades</h4>
            <div className="result-distribution__chart">
              {probabilityEntries.map(([label, score]) => (
                <div key={label} className="result-distribution__item">
                  <div className="result-distribution__header">
                    <span>{label}</span>
                    <strong>{formatPercent(score)}</strong>
                  </div>
                  <div className="result-distribution__bar">
                    <div 
                      className="result-distribution__fill" 
                      style={{ width: `${Math.max(0, Math.min(100, Number(score) * 100))}%` }} 
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="result-metadata">
            <h4>Metadatos del audio</h4>
            <div className="result-metadata__grid">
              <div>
                <span>Archivo</span>
                <strong>{prediction.metadata?.filename || '-'}</strong>
              </div>
              <div>
                <span>Duración</span>
                <strong>{prediction.metadata?.duration_seconds || '-'} s</strong>
              </div>
              <div>
                <span>Sample rate</span>
                <strong>{prediction.metadata?.sample_rate || '-'}</strong>
              </div>
            </div>
          </div>
        </>
      ) : (
        <EmptyResults />
      )}
    </div>
  )
}