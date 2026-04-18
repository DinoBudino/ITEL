import React, { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { FaMicrophone, FaBolt, FaChartBar } from 'react-icons/fa'
import { AppSidebar } from '../components/shared/AppSidebar'
import { PipelineSection } from '../components/shared/PipelineSection'
import { useHealthCheck } from '../hooks/useHealthCheck'

export function HomePage() {
  const health = useHealthCheck()
  const heroTitle = 'Analiza muestras de voz para identificar el Trastorno Específico del Lenguaje'
  const [typedTitle, setTypedTitle] = useState('')

  useEffect(() => {
    let index = 0
    const intervalId = window.setInterval(() => {
      index += 1
      setTypedTitle(heroTitle.slice(0, index))
      if (index >= heroTitle.length) {
        window.clearInterval(intervalId)
      }
    }, 28)
    return () => window.clearInterval(intervalId)
  }, [])

  return (
    <div className="app-shell app-shell--home">
      <AppSidebar health={health} />
      <main className="workspace workspace--home">
        <section className="landing-hero" id="overview">
          <div className="landing-hero__content">
            <span className="eyebrow">Introducción</span>
            
            <div className="hero-main">
              <div className="hero-text">
                <h2 className="landing-hero__title">
                  <span className="typewriter-text">{typedTitle}</span>
                  <span className="typewriter-caret" aria-hidden="true" />
                </h2>
                <p className="landing-hero__description">
                  ITEL procesa archivos de audio WAV mediante técnicas de aprendizaje automático para estimar la probabilidad de TEL, mostrando resultados con confianza.
                </p>
              </div>
              <div className="hero-action">
                <Link className="button button--primary button--large" to="/analysis">
                  Iniciar análisis
                </Link>
              </div>
            </div>

            <div className="hero-features">
              <div className="hero-feature">
                <FaMicrophone size={24} className="hero-feature__icon" />
                <div>
                  <strong>Formato de audio</strong>
                  <p>Archivos WAV mono a 16kHz</p>
                </div>
              </div>
              <div className="hero-feature">
                <FaBolt size={24} className="hero-feature__icon" />
                <div>
                  <strong>Procesamiento rápido</strong>
                  <p>Análisis en ~1.2 segundos</p>
                </div>
              </div>
              <div className="hero-feature">
                <FaChartBar size={24} className="hero-feature__icon" />
                <div>
                  <strong>Resultados claros</strong>
                  <p>Clase, confianza y métricas</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        <PipelineSection />
      </main>
    </div>
  )
}