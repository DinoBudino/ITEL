import React, { useEffect, useState } from 'react'

import { Route, Routes, useLocation } from 'react-router-dom'

import { AnalysisPage } from './pages/AnalysisPage'
import { HomePage } from './pages/HomePage'

function AnimatedRoutes() {
  const location = useLocation()
  const [displayLocation, setDisplayLocation] = useState(location)
  const [transitionStage, setTransitionStage] = useState('route-enter')

  useEffect(() => {
    if (location.pathname !== displayLocation.pathname) {
      setTransitionStage('route-exit')
    }
  }, [location, displayLocation])

  useEffect(() => {
    if (transitionStage !== 'route-exit') {
      return undefined
    }

    const timeoutId = window.setTimeout(() => {
      setDisplayLocation(location)
      setTransitionStage('route-enter')
    }, 180)

    return () => window.clearTimeout(timeoutId)
  }, [location, transitionStage])

  return (
    <div className={`route-shell ${transitionStage}`}>
      <Routes location={displayLocation}>
        <Route path="/" element={<HomePage />} />
        <Route path="/analysis" element={<AnalysisPage />} />
      </Routes>
    </div>
  )
}

function App() {
  return <AnimatedRoutes />
}

export default App
