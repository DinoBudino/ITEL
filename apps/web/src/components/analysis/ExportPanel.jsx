import React from 'react'
import { jsPDF } from 'jspdf'
import { FaFilePdf, FaFileCode, FaFileAlt, FaDownload } from 'react-icons/fa'
import { formatPercent, formatMs } from '../../utils/formatters'

export function ExportPanel({ prediction, probabilityEntries, selectedFile }) {
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

    doc.setFontSize(14)
    doc.text('Metadatos del Audio', 20, y)
    y += 10

    doc.setFontSize(11)
    doc.text(`Archivo: ${prediction.metadata?.filename || '-'}`, 20, y)
    y += 7
    doc.text(`Duración: ${prediction.metadata?.duration_seconds || '-'} s`, 20, y)
    y += 7
    doc.text(`Sample rate: ${prediction.metadata?.sample_rate || '-'}`, 20, y)
    y += 7
    doc.text(`Modelo: ${prediction.metadata?.model_version || 'itelv5'}`, 20, y)
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
      metadata: {
        ...prediction.metadata,
        model_version: prediction.metadata?.model_version || 'itelv5',
      },
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
      ['Nivel de confianza', prediction.confidence >= 0.85 ? 'Alta' : prediction.confidence >= 0.65 ? 'Media' : 'Exploratoria'],
      ...probabilityEntries?.map(([label, score]) => [label, score]) || [],
      ['Archivo', prediction.metadata?.filename || '-'],
      ['Duración (s)', prediction.metadata?.duration_seconds || '-'],
      ['Sample rate', prediction.metadata?.sample_rate || '-'],
      ['Modelo', prediction.metadata?.model_version || 'itelv5'],
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
      <div className="analysis-card__header">
        <span className="eyebrow">Exportar</span>
        <h3>Descargar informe</h3>
      </div>
      <p className="export-panel__copy">Genera un informe listo para compartir o descarga los datos crudos del analisis.</p>
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
