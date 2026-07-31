import React, { useEffect, useState } from 'react'
import { FaCheck, FaPlay, FaSearch, FaTable, FaTimes, FaUpload } from 'react-icons/fa'

import { statusLabels } from '../../constants/ui'

function SamplePagination({ samplePagination, onSamplePageChange, onSamplePageSizeChange }) {
  const { page, pageSize, total, totalPages } = samplePagination
  const rangeStart = total === 0 ? 0 : (page - 1) * pageSize + 1
  const rangeEnd = total === 0 ? 0 : Math.min(page * pageSize, total)

  return (
    <div className="sample-pagination">
      <span className="sample-pagination__summary">
        Mostrando {rangeStart}-{rangeEnd} de {total} muestras
      </span>

      <div className="sample-pagination__controls">
        <label className="sample-pagination__size">
          <span>Filas</span>
          <select value={pageSize} onChange={(event) => onSamplePageSizeChange(Number(event.target.value))}>
            <option value={5}>5</option>
            <option value={25}>25</option>
            <option value={50}>50</option>
            <option value={100}>100</option>
          </select>
        </label>

        <button className="button button--ghost" onClick={() => onSamplePageChange(page - 1)} disabled={page <= 1}>
          Anterior
        </button>
        <span className="sample-pagination__page">
          Pagina {page} de {Math.max(totalPages, 1)}
        </span>
        <button className="button button--ghost" onClick={() => onSamplePageChange(page + 1)} disabled={page >= totalPages || totalPages === 0}>
          Siguiente
        </button>
      </div>
    </div>
  )
}

function SampleAudioTable({ selectedSample, sampleAudios, sampleState, onSampleSelect }) {
  if (sampleState === 'loading') {
    return <div className="sample-table__empty">Cargando muestras LANNA...</div>
  }

  if (sampleAudios.length === 0) {
    return <div className="sample-table__empty">No se encontraron muestras con esos filtros.</div>
  }

  return (
    <div className="sample-table__wrapper">
      <table className="sample-table">
        <thead>
          <tr>
            <th>Clase</th>
            <th>Sujeto</th>
            <th>Tarea</th>
            <th>Archivo</th>
            <th>Accion</th>
          </tr>
        </thead>
        <tbody>
          {sampleAudios.map((sample) => {
            const isSelected = selectedSample?.id === sample.id

            return (
              <tr key={sample.id} className={isSelected ? 'is-selected' : ''}>
                <td>
                  <span className={`sample-class sample-class--${sample.class_label.toLowerCase()}`}>{sample.class_label}</span>
                </td>
                <td>{sample.subject_id}</td>
                <td>{sample.task_code}</td>
                <td className="sample-table__filename">{sample.filename}</td>
                <td className="sample-table__action-cell">
                  <button
                    className={`sample-table__action ${isSelected ? 'is-selected' : ''}`}
                    onClick={() => onSampleSelect(sample)}
                    type="button"
                    aria-label={isSelected ? `Muestra ${sample.filename} seleccionada` : `Usar muestra ${sample.filename}`}
                    title={isSelected ? 'Seleccionada' : 'Usar muestra'}
                  >
                    <FaCheck size={14} />
                  </button>
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

export function ProcessingPanel({ progress, onCancel }) {
  const [elapsedMs, setElapsedMs] = useState(0)

  useEffect(() => {
    const updateElapsed = () => setElapsedMs(Date.now() - progress.startedAt)
    updateElapsed()
    const timer = window.setInterval(updateElapsed, 250)
    return () => window.clearInterval(timer)
  }, [progress.startedAt])

  const percent = typeof progress.progress === 'number' ? Math.max(0, Math.min(100, progress.progress)) : null
  const isIndeterminate = progress.indeterminate || percent == null
  const displayedElapsed = Math.max(elapsedMs, progress.elapsed_ms || 0)
  const cacheKnown = progress.cache_hits != null || progress.cache_misses != null

  return (
    <div className="processing-panel" aria-live="polite">
      <div className="processing-panel__heading">
        <div>
          <span className="processing-panel__stage">{progress.stage || 'Procesando'}</span>
          <strong>{progress.message || 'Procesando grabaciones...'}</strong>
        </div>
        <span className="processing-panel__percent">{isIndeterminate ? 'En curso' : `${Math.round(percent)}%`}</span>
      </div>

      <div
        className={`processing-bar${isIndeterminate ? ' processing-bar--indeterminate' : ''}`}
        role="progressbar"
        aria-label="Progreso de la inferencia"
        aria-valuemin={isIndeterminate ? undefined : 0}
        aria-valuemax={isIndeterminate ? undefined : 100}
        aria-valuenow={isIndeterminate ? undefined : percent}
      >
        <span style={isIndeterminate ? undefined : { width: `${percent}%` }} />
      </div>

      <div className="processing-panel__meta">
        <span>Archivos: {progress.file_count == null ? 'Pendiente' : `${progress.file_index ?? 'Pendiente'} / ${progress.file_count}`}</span>
        <span>Tiempo: {(displayedElapsed / 1000).toFixed(1)} s</span>
        <span>Cache: {cacheKnown ? `${progress.cache_hits || 0} hits / ${progress.cache_misses || 0} misses` : 'Pendiente'}</span>
      </div>

      <button className="button button--ghost processing-panel__cancel" type="button" onClick={onCancel}>
        <FaTimes /> Cancelar
      </button>
    </div>
  )
}

export function PredictionForm({
  inputMode,
  selectedFiles,
  selectedSample,
  models,
  modelState,
  selectedModelId,
  selectedLanguage,
  availableLanguages,
  modelAvailable,
  audioUrl,
  predictionState,
  streamProgress,
  errorMessage,
  onInputModeChange,
  onFileChange,
  onModelChange,
  onLanguageChange,
  onPredict,
  onCancel,
  onClear,
}) {
  const hasSelection = inputMode === 'upload' ? selectedFiles.length > 0 : Boolean(selectedSample)
  const selectedLabel = inputMode === 'upload'
    ? `${selectedFiles.length} archivo${selectedFiles.length === 1 ? '' : 's'} WAV`
    : selectedSample?.filename

  return (
    <div className="analysis-card analysis-upload" id="sample-upload">
      <div className="analysis-card__header">
        <span className="eyebrow">Entrada</span>
        <h3>Analizar grabaciones de un sujeto</h3>
      </div>

      <div className="prediction-options">
        <label className="prediction-options__field">
          <span>Modelo</span>
          <select value={selectedModelId} onChange={(event) => onModelChange(event.target.value)} disabled={modelState !== 'ready' || predictionState === 'processing'}>
            {models.map((model) => (
              <option key={model.id} value={model.id}>{model.name}{model.available ? '' : ' (artefactos pendientes)'}</option>
            ))}
          </select>
        </label>

        <label className="prediction-options__field">
          <span>Idioma</span>
          <select
            value={selectedLanguage}
            onChange={(event) => onLanguageChange(event.target.value)}
            disabled={modelState !== 'ready' || availableLanguages.length === 0 || predictionState === 'processing'}
          >
            {availableLanguages.map((language) => (
              <option key={language} value={language}>{language.toUpperCase()}</option>
            ))}
          </select>
        </label>
      </div>

      {modelState === 'loading' && <div className="model-catalog-note">Consultando modelos disponibles...</div>}
      {modelState === 'error' && (
        <div className="model-catalog-note model-catalog-note--error">
          Catálogo no disponible. La selección se muestra como referencia, pero el análisis requiere conexión con la API.
        </div>
      )}
      {modelState === 'ready' && !modelAvailable && (
        <div className="model-catalog-note model-catalog-note--error">
          Este modelo está registrado, pero su bundle no está disponible o no pasó la validación de integridad.
        </div>
      )}

      {inputMode === 'upload' ? (
        <label className="upload-zone">
          <input type="file" accept=".wav,audio/wav" multiple onChange={onFileChange} className="upload-zone__input" disabled={predictionState === 'processing'} />
          <div className="upload-zone__content">
            <FaUpload size={28} className="upload-zone__icon" />
            <span className="upload-zone__text">
              {selectedFiles.length > 0 ? selectedFiles.map((file) => file.name).join(', ') : 'Seleccionar uno o más archivos WAV'}
            </span>
          </div>
        </label>
      ) : (
        <div className="sample-browser-hint">
          <FaTable />
          <span>Selecciona una muestra en la tabla LANNA que aparece debajo del panel principal.</span>
        </div>
      )}

      {selectedLabel && <div className="analysis-selection">Seleccion actual: <strong>{selectedLabel}</strong></div>}

      {audioUrl && <audio controls src={audioUrl} className="audio-player" />}

      {predictionState === 'processing' && streamProgress && (
        <ProcessingPanel progress={streamProgress} onCancel={onCancel} />
      )}

      <div className="analysis-actions">
        <button
          className="button button--primary"
          onClick={onPredict}
          disabled={predictionState === 'processing' || !hasSelection || modelState !== 'ready' || !modelAvailable || !selectedLanguage}
        >
          <FaPlay /> Analizar
        </button>
        {hasSelection && predictionState !== 'processing' && (
          <button className="button button--ghost" onClick={onClear}>
            <FaTimes />
          </button>
        )}
      </div>

      <div className="analysis-status">
        <span className={`status-dot status-dot--${predictionState}`} />
        <span>{statusLabels[predictionState]}</span>
      </div>

      {errorMessage && <div className="analysis-error">{errorMessage}</div>}
    </div>
  )
}

export function SampleBrowserPanel({
  selectedSample,
  sampleAudios,
  sampleFilters,
  samplePagination,
  sampleState,
  onSampleSelect,
  onSampleFilterChange,
  onSamplePageChange,
  onSamplePageSizeChange,
}) {
  return (
    <div className="analysis-card sample-browser-panel">
      <div className="analysis-card__header">
        <span className="eyebrow">LANNA</span>
        <h3>Biblioteca de muestras</h3>
      </div>

      <div className="sample-browser">
        <div className="sample-browser__note">Muestras de prueba provenientes de LANNA Speech Database.</div>

        <div className="sample-filters">
          <label className="sample-filters__field">
            <span>Clase</span>
            <select value={sampleFilters.classLabel} onChange={(event) => onSampleFilterChange('classLabel', event.target.value)}>
              <option value="All">Todas</option>
              <option value="Healthy">Healthy</option>
              <option value="Patients">Patients</option>
            </select>
          </label>

          <label className="sample-filters__field sample-filters__field--search">
            <span>Buscar</span>
            <div className="sample-search">
              <FaSearch />
              <input
                type="search"
                value={sampleFilters.query}
                onChange={(event) => onSampleFilterChange('query', event.target.value)}
                placeholder="Archivo, sujeto o tarea"
              />
            </div>
          </label>
        </div>

        <SampleAudioTable selectedSample={selectedSample} sampleAudios={sampleAudios} sampleState={sampleState} onSampleSelect={onSampleSelect} />

        <SamplePagination
          samplePagination={samplePagination}
          onSamplePageChange={onSamplePageChange}
          onSamplePageSizeChange={onSamplePageSizeChange}
        />
      </div>
    </div>
  )
}
