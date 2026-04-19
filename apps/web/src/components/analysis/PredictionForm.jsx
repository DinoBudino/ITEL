import React from 'react'
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

export function PredictionForm({
  inputMode,
  selectedFile,
  selectedSample,
  audioUrl,
  predictionState,
  errorMessage,
  onInputModeChange,
  onFileChange,
  onPredict,
  onClear,
}) {
  const hasSelection = inputMode === 'upload' ? Boolean(selectedFile) : Boolean(selectedSample)
  const selectedLabel = inputMode === 'upload' ? selectedFile?.name : selectedSample?.filename

  return (
    <div className="analysis-card analysis-upload" id="sample-upload">
      <div className="analysis-card__header">
        <span className="eyebrow">Entrada</span>
        <h3>Subir audio o elegir de LANNA</h3>
      </div>

      <div className="input-mode-tabs" role="tablist" aria-label="Modo de entrada">
        <button
          className={`input-mode-tab ${inputMode === 'upload' ? 'is-active' : ''}`}
          onClick={() => onInputModeChange('upload')}
          type="button"
        >
          <FaUpload /> Subir audio
        </button>
        <button
          className={`input-mode-tab ${inputMode === 'samples' ? 'is-active' : ''}`}
          onClick={() => onInputModeChange('samples')}
          type="button"
        >
          <FaTable /> Elegir de LANNA
        </button>
      </div>

      {inputMode === 'upload' ? (
        <label className="upload-zone">
          <input type="file" accept=".wav,audio/wav" onChange={onFileChange} className="upload-zone__input" />
          <div className="upload-zone__content">
            <FaUpload size={28} className="upload-zone__icon" />
            <span className="upload-zone__text">{selectedFile ? selectedFile.name : 'Seleccionar archivo WAV'}</span>
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

      <div className="analysis-actions">
        <button className="button button--primary" onClick={onPredict} disabled={predictionState === 'processing' || !hasSelection}>
          <FaPlay /> Analizar
        </button>
        {hasSelection && (
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
