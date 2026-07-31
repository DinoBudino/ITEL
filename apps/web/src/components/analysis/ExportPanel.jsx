import React from 'react'
import { jsPDF } from 'jspdf'
import { FaFilePdf, FaFileCode, FaFileAlt } from 'react-icons/fa'
import { formatPercent } from '../../utils/formatters'

export function ExportPanel({ prediction, probabilityEntries }) {
  const handleExportPDF = () => {
    if (!prediction) return

    const doc = new jsPDF()
    const pageWidth = doc.internal.pageSize.getWidth()
    let y = 20

    doc.setFontSize(18)
    doc.setTextColor(37, 99, 235)
    doc.text('Informe de Análisis ITEL', pageWidth / 2, y, { align: 'center' })
    y += 15

    doc.setFontSize(12)
    doc.setTextColor(0, 0, 0)
    doc.text('Identificación del Trastorno Específico del Lenguaje', pageWidth / 2, y, { align: 'center' })
    y += 20

    doc.setFontSize(14)
    doc.text('Resultados del Análisis', 20, y)
    y += 10

    doc.setFontSize(11)
    doc.text(`Predicción: ${prediction.predicted_label}`, 20, y)
    y += 8
    doc.text(`Confianza: ${formatPercent(prediction.confidence)}`, 20, y)
    y += 15

    doc.setFontSize(14)
    doc.text('Probabilidades', 20, y)
    y += 10

    doc.setFontSize(11)
    probabilityEntries?.forEach(([label, score]) => {
      doc.text(`${label}: ${formatPercent(score)}`, 20, y)
      y += 7
    })
    y += 10

    doc.setFontSize(11)
    doc.text(`Modelo: ${prediction.model_id}`, 20, y)
    y += 7
    doc.text(`Idioma: ${prediction.language}`, 20, y)
    y += 7
    doc.text(`Archivos analizados: ${prediction.file_count}`, 20, y)
    if (prediction.reference_label != null) {
      y += 7
      doc.text(`Sujeto demo: ${prediction.demo_subject_display_id}`, 20, y)
      y += 7
      doc.text(`Referencia: ${prediction.reference_label} (${prediction.reference_match ? 'coincide' : 'no coincide'})`, 20, y)
    }
    y += 15

    doc.setFontSize(10)
    doc.setTextColor(100, 100, 100)
    doc.text('Nota: Este resultado es una estimación basada en análisis acústico automatizado.', 20, y)
    y += 6
    doc.text('Debe ser interpretado por un profesional de la salud.', 20, y)
    y += 6
    doc.text(`Fecha: ${new Date().toLocaleString('es-MX')}`, 20, y)

    doc.save('itel-informe.pdf')
  }

  const handleExportJSON = () => {
    if (!prediction) return

    const data = {
      prediction: {
        label: prediction.predicted_label,
        confidence: prediction.confidence,
        confidence_percent: formatPercent(prediction.confidence),
      },
      probabilities: Object.fromEntries(probabilityEntries || []),
      model_id: prediction.model_id,
      model_version: prediction.model_version,
      model_digest: prediction.model_digest,
      language: prediction.language,
      file_count: prediction.file_count,
      threshold: prediction.threshold,
      feature_source: prediction.feature_source,
      cache_hits: prediction.cache_hits,
      cache_misses: prediction.cache_misses,
      device: prediction.device,
      demo_subject_display_id: prediction.demo_subject_display_id,
      reference_label: prediction.reference_label,
      reference_match: prediction.reference_match,
      warnings: prediction.warnings,
      timings_ms: prediction.timings_ms,
      generated_at: new Date().toISOString(),
    }

    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'itel-resultado.json'
    a.click()
    URL.revokeObjectURL(url)
  }

  const handleExportCSV = () => {
    if (!prediction) return

    const rows = [
      ['Campo', 'Valor'],
      ['Predicción', prediction.predicted_label],
      ['Confianza', prediction.confidence],
      ['Confianza %', formatPercent(prediction.confidence)],
      ...probabilityEntries?.map(([label, score]) => [label, score]) || [],
      ['Modelo', prediction.model_id],
      ['Versión', prediction.model_version],
      ['Digest', prediction.model_digest],
      ['Idioma', prediction.language],
      ['Archivos', prediction.file_count],
      ['Threshold', prediction.threshold],
      ['Fuente de features', prediction.feature_source],
      ['Cache hits', prediction.cache_hits],
      ['Cache misses', prediction.cache_misses],
      ['Dispositivo', prediction.device],
      ...(prediction.reference_label != null ? [
        ['Sujeto demo', prediction.demo_subject_display_id],
        ['Referencia', prediction.reference_label],
        ['Coincidencia', prediction.reference_match ? 'Sí' : 'No'],
      ] : []),
      ['Tiempo total (ms)', prediction.timings_ms?.total || '-'],
      ['Fecha', new Date().toISOString()],
    ]

    const csv = rows.map(row => row.join(',')).join('\n')
    const blob = new Blob([csv], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'itel-resultado.csv'
    a.click()
    URL.revokeObjectURL(url)
  }

  if (!prediction) return null

  return (
    <div className="analysis-card export-panel">
      <div className="export-toolbar__header">
        <span className="eyebrow">Exportar</span>
      </div>
      <div className="export-options">
        <button className="export-button" onClick={handleExportPDF}>
          <FaFilePdf size={20} />
          <span>PDF</span>
        </button>
        <button className="export-button" onClick={handleExportJSON}>
          <FaFileCode size={20} />
          <span>JSON</span>
        </button>
        <button className="export-button" onClick={handleExportCSV}>
          <FaFileAlt size={20} />
          <span>CSV</span>
        </button>
      </div>
    </div>
  )
}
