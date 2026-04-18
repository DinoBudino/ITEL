import React from 'react'
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from 'chart.js'
import { Doughnut } from 'react-chartjs-2'
import { formatPercent } from '../../utils/formatters'

ChartJS.register(ArcElement, Tooltip, Legend)

export function ProbabilityChart({ probabilityEntries, prediction }) {
  if (!probabilityEntries || probabilityEntries.length === 0) return null

  const labels = probabilityEntries.map(([label]) => label)
  const scores = probabilityEntries.map(([, score]) => Number(score))

  const data = {
    labels,
    datasets: [
      {
        data: scores,
        backgroundColor: ['#2563eb', '#93c5fd'],
        borderColor: ['#ffffff', '#ffffff'],
        borderWidth: 2,
      },
    ],
  }

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'bottom',
        labels: {
          padding: 16,
          usePointStyle: true,
          font: {
            size: 12,
            family: "'Public Sans', sans-serif",
          },
        },
      },
      tooltip: {
        callbacks: {
          label: (context) => `${context.label}: ${formatPercent(context.raw)}`,
        },
      },
    },
    cutout: '65%',
  }

  const predictedLabel = prediction?.predicted_label

  return (
    <div className="analysis-card probability-chart">
      <div className="analysis-card__header">
        <span className="eyebrow">Distribución</span>
        <h3>Probabilidades</h3>
      </div>
      <div className="chart-container">
        <Doughnut data={data} options={options} />
      </div>
      {predictedLabel && (
        <div className="chart-legend">
          <span className="chart-predicted">
            Predicción: <strong>{predictedLabel}</strong>
          </span>
        </div>
      )}
    </div>
  )
}