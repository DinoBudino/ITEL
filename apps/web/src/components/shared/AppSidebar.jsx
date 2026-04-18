import React from 'react'
import { Link, NavLink, useLocation } from 'react-router-dom'

const sidebarSections = {
  home: {
    ctaTo: '/analysis',
    ctaLabel: 'Iniciar análisis'
  },
  analysis: {
    ctaTo: '/',
    ctaLabel: 'Volver al inicio'
  }
}

export function AppSidebar({ health }) {
  const location = useLocation()
  const isAnalysisView = location.pathname === '/analysis'
  const currentSection = isAnalysisView ? sidebarSections.analysis : sidebarSections.home

  return (
    <aside className="sidebar">
      <div className="brand-block brand-block--sidebar">
        <div className="brand-mark">ITEL</div>
        <h1>Identificación del Trastorno Específico del Lenguaje</h1>
      </div>

      <nav className="sidebar-nav">
        <NavLink to="/" className={({ isActive }) => (isActive ? 'is-active' : '')} end>
          Inicio
        </NavLink>
      </nav>

      <div className="sidebar-card">
        <span>Estado del servicio</span>
        <strong className={health.status === 'ready' ? 'text-success' : health.status === 'error' ? 'text-danger' : ''}>
          {health.message}
        </strong>
        <small>{health.modelVersion ? `Modelo ${health.modelVersion}` : 'Versión de modelo pendiente'}</small>
      </div>

      <Link className={`button ${isAnalysisView ? 'button--outline' : 'button--primary'} sidebar-cta`} to={currentSection.ctaTo}>
        {currentSection.ctaLabel}
      </Link>
    </aside>
  )
}
