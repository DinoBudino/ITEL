import React from 'react'
import { FaCheck, FaClock, FaFileAudio, FaGlobe, FaPlay, FaTimes, FaUser } from 'react-icons/fa'

import { API_BASE_URL } from '../../constants/api'
import { statusLabels } from '../../constants/ui'
import { ProcessingPanel } from './PredictionForm'

function formatDuration(seconds) {
  const value = Math.max(0, Number(seconds) || 0)
  const minutes = Math.floor(value / 60)
  const remainder = Math.round(value % 60)
  return minutes ? `${minutes} min ${remainder} s` : `${remainder} s`
}

function streamSource(streamUrl) {
  if (!streamUrl || /^https?:\/\//i.test(streamUrl)) return streamUrl
  if (API_BASE_URL.startsWith('/') && streamUrl.startsWith(`${API_BASE_URL}/`)) return streamUrl
  return `${API_BASE_URL.replace(/\/$/, '')}/${streamUrl.replace(/^\//, '')}`
}

export function DemoLibraryPanel({ models, modelState, demo }) {
  const { page, pageSize, total, totalPages } = demo.pagination
  const first = total ? (page - 1) * pageSize + 1 : 0
  const last = Math.min(page * pageSize, total)

  return (
    <div className="analysis-card demo-library">
      <div className="analysis-card__header">
        <span className="eyebrow">Biblioteca privada</span>
        <h3>Base de pruebas por sujeto</h3>
      </div>

      <div className="prediction-options">
        <label className="prediction-options__field">
          <span>Modelo</span>
          <select value={demo.selectedModelId} onChange={(event) => demo.handleModelChange(event.target.value)} disabled={modelState !== 'ready' || demo.predictionState === 'processing'}>
            {models.map((model) => <option key={model.id} value={model.id}>{model.name}{model.available ? '' : ' (artefactos pendientes)'}</option>)}
          </select>
        </label>
        <label className="prediction-options__field">
          <span>Idioma {demo.selectedSubject ? '(fijado por el sujeto)' : ''}</span>
          <select value={demo.selectedLanguage} onChange={(event) => demo.handleLanguageChange(event.target.value)} disabled={modelState !== 'ready' || demo.availableLanguages.length === 0 || Boolean(demo.selectedSubject) || demo.predictionState === 'processing'}>
            {demo.availableLanguages.map((language) => <option key={language} value={language}>{language.toUpperCase()}</option>)}
          </select>
        </label>
        <label className="prediction-options__field">
          <span>Clasificación</span>
          <select value={demo.selectedReferenceLabel} onChange={(event) => demo.handleReferenceLabelChange(event.target.value)} disabled={demo.predictionState === 'processing'}>
            <option value="">Todas</option>
            <option value="Healthy">Healthy</option>
            <option value="Patients">Patients</option>
          </select>
        </label>
      </div>

      {!demo.modelAvailable && modelState === 'ready' && <div className="model-catalog-note model-catalog-note--error">El bundle de este modelo no está disponible.</div>}

      <div className="demo-subject-grid" aria-busy={demo.libraryState === 'loading'}>
        {demo.libraryState === 'loading' && <div className="sample-table__empty">Cargando sujetos de prueba...</div>}
        {demo.libraryState !== 'loading' && demo.subjects.length === 0 && <div className="sample-table__empty">No hay sujetos disponibles para esta selección.</div>}
        {demo.subjects.map((subject) => {
          const selected = demo.selectedSubject?.token === subject.token
          return (
            <button key={subject.token} type="button" className={`demo-subject${selected ? ' is-selected' : ''}`} onClick={() => demo.handleSubjectSelect(subject)} disabled={demo.predictionState === 'processing'}>
              <span className="demo-subject__identity"><FaUser /><strong>{subject.display_id}</strong></span>
              <span className={`demo-reference-badge demo-reference-badge--${subject.reference_label.toLowerCase()}`}>{subject.reference_label}</span>
              <span><FaGlobe /> {subject.language.toUpperCase()}</span>
              <span><FaFileAudio /> {subject.audio_count} grabaciones</span>
              <span><FaClock /> {formatDuration(subject.total_duration_seconds)}</span>
              <span className="demo-subject__tasks">{(subject.task_types || []).join(' · ') || 'Tareas sin especificar'}</span>
              <span className="demo-subject__select"><FaCheck /> {selected ? 'Seleccionado' : 'Seleccionar'}</span>
            </button>
          )
        })}
      </div>

      <div className="sample-pagination">
        <span className="sample-pagination__summary">Mostrando {first}-{last} de {total} sujetos</span>
        <div className="sample-pagination__controls">
          <button type="button" className="button button--ghost" onClick={() => demo.handlePageChange(page - 1)} disabled={page <= 1}>Anterior</button>
          <span className="sample-pagination__page">Página {page} de {Math.max(totalPages, 1)}</span>
          <button type="button" className="button button--ghost" onClick={() => demo.handlePageChange(page + 1)} disabled={!totalPages || page >= totalPages}>Siguiente</button>
        </div>
      </div>

      {demo.selectedSubject && (
        <section className="demo-recordings">
          <div className="demo-recordings__header">
            <div>
              <span className="eyebrow">Sujeto seleccionado</span>
              <div className="demo-recordings__identity">
                <h4>{demo.selectedSubject.display_id}</h4>
                <span className={`demo-reference-badge demo-reference-badge--${demo.selectedSubject.reference_label.toLowerCase()}`}>{demo.selectedSubject.reference_label}</span>
              </div>
            </div>
            <button type="button" className="button button--ghost" onClick={demo.handleClearSubject} disabled={demo.predictionState === 'processing'} aria-label="Quitar selección"><FaTimes /></button>
          </div>
          <div className="demo-recordings__list">
            {(demo.selectedSubject.audios || []).map((audio) => (
              <article className="demo-recording" key={audio.token}>
                <div><strong>{audio.display_name}</strong><span>{audio.task_type} · {formatDuration(audio.duration_seconds)}</span></div>
                <audio controls preload="none" src={streamSource(audio.stream_url)} aria-label={`Reproducir ${audio.display_name}`} />
              </article>
            ))}
          </div>
        </section>
      )}

      {demo.predictionState === 'processing' && demo.streamProgress && <ProcessingPanel progress={demo.streamProgress} onCancel={demo.handleCancel} />}

      <div className="analysis-actions">
        <button type="button" className="button button--primary" onClick={demo.handlePredict} disabled={!demo.selectedSubject || !demo.modelAvailable || demo.predictionState === 'processing'}><FaPlay /> Analizar sujeto</button>
      </div>
      <div className="analysis-status"><span className={`status-dot status-dot--${demo.predictionState}`} /><span>{statusLabels[demo.predictionState]}</span></div>
      {demo.errorMessage && <div className="analysis-error">{demo.errorMessage}</div>}
    </div>
  )
}
