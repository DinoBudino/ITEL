import React from 'react'
import { FaCheckCircle, FaClock, FaUpload, FaPlay, FaFileAudio, FaBrain, FaGlobe, FaDatabase } from 'react-icons/fa'
import { formatMs, formatPercent } from '../../utils/formatters'

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
            <p>Selecciona uno o más archivos WAV de voz</p>
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
              Confianza: {formatPercent(prediction.confidence)}
            </p>
            <p className="result-main__description">Resultado experimental a nivel de sujeto; no constituye un diagnóstico.</p>
          </div>

          {prediction.reference_label != null && (
            <div className={`reference-result${prediction.reference_match ? ' is-match' : ' is-mismatch'}`}>
              <span className="eyebrow">Referencia de la base de pruebas</span>
              <div className="reference-result__line">
                <strong>{prediction.demo_subject_display_id}</strong>
                <span>Referencia: {prediction.reference_label}</span>
                <span>{prediction.reference_match ? 'Coincide con la predicción' : 'No coincide con la predicción'}</span>
              </div>
            </div>
          )}

          <div className="result-metrics">
            <div className="result-metrics__item">
              <FaBrain size={18} />
              <span className="result-metrics__label">Modelo</span>
              <strong className="result-metrics__value">{prediction.model_id}</strong>
            </div>
            <div className="result-metrics__item">
              <FaGlobe size={18} />
              <span className="result-metrics__label">Idioma</span>
              <strong className="result-metrics__value">{prediction.language?.toUpperCase()}</strong>
            </div>
            <div className="result-metrics__item">
              <FaDatabase size={18} />
              <span className="result-metrics__label">Archivos</span>
              <strong className="result-metrics__value">{prediction.file_count}</strong>
            </div>
          </div>

          {prediction.warnings?.length > 0 && (
            <div className="analysis-error">{prediction.warnings.join(' ')}</div>
          )}

          <div className="result-metrics result-metrics--single">
            <div className="result-metrics__item">
              <FaDatabase size={18} />
              <span className="result-metrics__label">Fuente de features</span>
              <strong className="result-metrics__value">
                {prediction.feature_source || 'computed'} · {prediction.cache_hits || 0} cache hits
              </strong>
            </div>
          </div>

          {prediction.timings_ms?.total != null && (
            <div className="result-metrics result-metrics--single">
              <div className="result-metrics__item">
                <FaClock size={18} />
                <span className="result-metrics__label">Tiempo total reportado</span>
                <strong className="result-metrics__value">{formatMs(prediction.timings_ms.total)}</strong>
              </div>
            </div>
          )}

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

        </>
      ) : (
        <EmptyResults />
      )}
    </div>
  )
}
