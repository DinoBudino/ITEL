import React, { useEffect, useState } from 'react'
import { AnalysisHeader } from '../components/analysis/AnalysisHeader'
import { PredictionForm } from '../components/analysis/PredictionForm'
import { ResultsPanel } from '../components/analysis/ResultsPanel'
import { AudioVisualizer } from '../components/analysis/AudioVisualizer'
import { ProbabilityChart } from '../components/analysis/ProbabilityChart'
import { ExportPanel } from '../components/analysis/ExportPanel'
import { DemoLibraryPanel } from '../components/analysis/DemoLibraryPanel'
import { AppSidebar } from '../components/shared/AppSidebar'
import { useAudioPreview } from '../hooks/useAudioPreview'
import { useHealthCheck } from '../hooks/useHealthCheck'
import { useModels } from '../hooks/useModels'
import { usePrediction } from '../hooks/usePrediction'
import { useDemoCapability } from '../hooks/useDemoCapability'
import { useDemoLibrary } from '../hooks/useDemoLibrary'

export function AnalysisPage() {
  const health = useHealthCheck()
  const { models, modelState } = useModels()
  const demoLibraryEnabled = useDemoCapability()
  const [workspaceMode, setWorkspaceMode] = useState('upload')
  const {
    inputMode,
    selectedFiles,
    selectedSample,
    selectedModelId,
    selectedLanguage,
    availableLanguages,
    modelAvailable,
    selectedSampleAudioUrl,
    predictionState,
    prediction,
    streamProgress,
    errorMessage,
    probabilityEntries,
    sampleAudios,
    sampleFilters,
    samplePagination,
    sampleState,
    handleInputModeChange,
    handleFileChange,
    handleModelChange,
    handleLanguageChange,
    handlePredict,
    handleCancel,
    handleSampleSelect,
    handleSampleFilterChange,
    handleSamplePageChange,
    handleSamplePageSizeChange,
    clearState,
  } =
    usePrediction(models, modelState)
  const uploadedAudioUrl = useAudioPreview(selectedFiles)
  const audioUrl = inputMode === 'samples' ? selectedSampleAudioUrl : uploadedAudioUrl
  const demo = useDemoLibrary(models, modelState, workspaceMode === 'demo')
  const activePrediction = workspaceMode === 'demo' ? demo.prediction : prediction
  const activeProbabilityEntries = workspaceMode === 'demo' ? demo.probabilityEntries : probabilityEntries
  const predictionInProgress = predictionState === 'processing' || demo.predictionState === 'processing'

  useEffect(() => {
    if (!demoLibraryEnabled) setWorkspaceMode('upload')
  }, [demoLibraryEnabled])

  return (
    <div className="app-shell app-shell--analysis">
      <AppSidebar health={health} />
      <main className="workspace workspace--analysis">
        <AnalysisHeader />

        <section className="analysis-workspace" id="dashboard">
          {demoLibraryEnabled && (
            <div className="input-mode-tabs" role="tablist" aria-label="Origen de las grabaciones">
              <button type="button" role="tab" aria-selected={workspaceMode === 'upload'} className={`input-mode-tab${workspaceMode === 'upload' ? ' is-active' : ''}`} onClick={() => setWorkspaceMode('upload')} disabled={predictionInProgress && workspaceMode !== 'upload'}>Subir archivo</button>
              <button type="button" role="tab" aria-selected={workspaceMode === 'demo'} className={`input-mode-tab${workspaceMode === 'demo' ? ' is-active' : ''}`} onClick={() => setWorkspaceMode('demo')} disabled={predictionInProgress && workspaceMode !== 'demo'}>Base de pruebas</button>
            </div>
          )}
          <div className="analysis-columns">
            <div className="analysis-left-column">
              {workspaceMode === 'upload' ? <PredictionForm
                inputMode={inputMode}
                selectedFiles={selectedFiles}
                selectedSample={selectedSample}
                models={models}
                modelState={modelState}
                selectedModelId={selectedModelId}
                selectedLanguage={selectedLanguage}
                availableLanguages={availableLanguages}
                modelAvailable={modelAvailable}
                audioUrl={audioUrl}
                predictionState={predictionState}
                streamProgress={streamProgress}
                errorMessage={errorMessage}
                onInputModeChange={handleInputModeChange}
                onFileChange={handleFileChange}
                onModelChange={handleModelChange}
                onLanguageChange={handleLanguageChange}
                onPredict={handlePredict}
                onCancel={handleCancel}
                onClear={clearState}
              /> : <DemoLibraryPanel models={models} modelState={modelState} demo={demo} />}

              {workspaceMode === 'upload' && <AudioVisualizer audioUrl={audioUrl} />}

            </div>

            <div className="analysis-right-column">
              <ExportPanel prediction={activePrediction} probabilityEntries={activeProbabilityEntries} />
              <ResultsPanel prediction={activePrediction} probabilityEntries={activeProbabilityEntries} />
              <ProbabilityChart probabilityEntries={activeProbabilityEntries} prediction={activePrediction} />
            </div>
          </div>
        </section>
      </main>
    </div>
  )
}
