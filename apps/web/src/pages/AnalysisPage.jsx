import React from 'react'
import { AnalysisHeader } from '../components/analysis/AnalysisHeader'
import { PredictionForm } from '../components/analysis/PredictionForm'
import { ResultsPanel } from '../components/analysis/ResultsPanel'
import { AudioVisualizer } from '../components/analysis/AudioVisualizer'
import { ProbabilityChart } from '../components/analysis/ProbabilityChart'
import { ExportPanel } from '../components/analysis/ExportPanel'
import { AppSidebar } from '../components/shared/AppSidebar'
import { useAudioPreview } from '../hooks/useAudioPreview'
import { useHealthCheck } from '../hooks/useHealthCheck'
import { usePrediction } from '../hooks/usePrediction'

export function AnalysisPage() {
  const health = useHealthCheck()
  const { selectedFile, predictionState, prediction, errorMessage, probabilityEntries, handleFileChange, handlePredict, clearState } =
    usePrediction()
  const audioUrl = useAudioPreview(selectedFile)

  return (
    <div className="app-shell app-shell--analysis">
      <AppSidebar health={health} />
      <main className="workspace workspace--analysis">
        <AnalysisHeader />

        <section className="analysis-workspace" id="dashboard">
          <div className="analysis-side-column">
            <PredictionForm
              selectedFile={selectedFile}
              audioUrl={audioUrl}
              predictionState={predictionState}
              errorMessage={errorMessage}
              onFileChange={handleFileChange}
              onPredict={handlePredict}
              onClear={clearState}
            />
            <AudioVisualizer audioUrl={audioUrl} />
          </div>

          <div className="analysis-main-column">
            <div className="analysis-results-layout">
              <div className="analysis-results-stack">
                <ResultsPanel prediction={prediction} probabilityEntries={probabilityEntries} />
                <ExportPanel prediction={prediction} probabilityEntries={probabilityEntries} selectedFile={selectedFile} />
              </div>
              <ProbabilityChart probabilityEntries={probabilityEntries} prediction={prediction} />
            </div>
          </div>
        </section>
      </main>
    </div>
  )
}
