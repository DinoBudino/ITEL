import React from 'react'
import { AnalysisHeader } from '../components/analysis/AnalysisHeader'
import { PredictionForm, SampleBrowserPanel } from '../components/analysis/PredictionForm'
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
  const {
    inputMode,
    selectedFile,
    selectedSample,
    selectedSampleAudioUrl,
    predictionState,
    prediction,
    errorMessage,
    probabilityEntries,
    sampleAudios,
    sampleFilters,
    samplePagination,
    sampleState,
    handleInputModeChange,
    handleFileChange,
    handlePredict,
    handleSampleSelect,
    handleSampleFilterChange,
    handleSamplePageChange,
    handleSamplePageSizeChange,
    clearState,
  } =
    usePrediction()
  const uploadedAudioUrl = useAudioPreview(selectedFile)
  const audioUrl = inputMode === 'samples' ? selectedSampleAudioUrl : uploadedAudioUrl

  return (
    <div className="app-shell app-shell--analysis">
      <AppSidebar health={health} />
      <main className="workspace workspace--analysis">
        <AnalysisHeader />

        <section className="analysis-workspace" id="dashboard">
          <div className="analysis-columns">
            <div className="analysis-left-column">
              <PredictionForm
                inputMode={inputMode}
                selectedFile={selectedFile}
                selectedSample={selectedSample}
                audioUrl={audioUrl}
                predictionState={predictionState}
                errorMessage={errorMessage}
                onInputModeChange={handleInputModeChange}
                onFileChange={handleFileChange}
                onPredict={handlePredict}
                onClear={clearState}
              />

              <AudioVisualizer audioUrl={audioUrl} />

              {inputMode === 'samples' && (
                <SampleBrowserPanel
                  selectedSample={selectedSample}
                  sampleAudios={sampleAudios}
                  sampleFilters={sampleFilters}
                  samplePagination={samplePagination}
                  sampleState={sampleState}
                  onSampleSelect={handleSampleSelect}
                  onSampleFilterChange={handleSampleFilterChange}
                  onSamplePageChange={handleSamplePageChange}
                  onSamplePageSizeChange={handleSamplePageSizeChange}
                />
              )}
            </div>

            <div className="analysis-right-column">
              <ExportPanel prediction={prediction} probabilityEntries={probabilityEntries} selectedFile={selectedFile} />
              <ResultsPanel prediction={prediction} probabilityEntries={probabilityEntries} />
              <ProbabilityChart probabilityEntries={probabilityEntries} prediction={prediction} />
            </div>
          </div>
        </section>
      </main>
    </div>
  )
}
