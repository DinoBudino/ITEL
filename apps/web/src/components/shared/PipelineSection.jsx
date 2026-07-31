// apps/web/src/components/shared/PipelineSection.jsx
import React, { useRef, useState } from 'react';
import { motion, useInView, AnimatePresence } from 'framer-motion';
import { 
  FaCheckCircle, 
  FaWaveSquare,
  FaNetworkWired,
  FaBalanceScale,
  FaClipboardList,
  FaChevronLeft,
  FaChevronRight,
  FaMicrophone,
  FaMagic
} from 'react-icons/fa';

const pipelineStepsData = [
  {
    title: 'Validación del audio',
    subtitle: 'Verificación inicial',
    description: 'Se verifica que cada archivo corresponda a una muestra de audio WAV válida antes de iniciar el análisis.',
    technicalNote: 'Si falla, se rechaza con mensaje claro para el usuario.',
    icon: FaMicrophone,
    color: '#3b82f6'
  },
  {
    title: 'Preprocesamiento',
    subtitle: 'Limpieza y normalización',
    description: 'Las grabaciones se preparan para producir una señal consistente y compatible con el modelo seleccionado.',
    technicalNote: 'El procesamiento se aplica de forma independiente a cada archivo.',
    icon: FaMagic,
    color: '#2563eb'
  },
  {
    title: 'Análisis acústico',
    subtitle: 'Modelo seleccionado',
    description: 'El modelo activo procesa cada grabación de acuerdo con el idioma indicado para obtener evidencia acústica.',
    technicalNote: 'La arquitectura y los idiomas compatibles se obtienen del catálogo de modelos.',
    icon: FaNetworkWired,
    color: '#1d4ed8'
  },
  {
    title: 'Agregación por sujeto',
    subtitle: 'Predicción conjunta',
    description: 'La evidencia de todos los archivos se combina para producir una única predicción a nivel de sujeto.',
    technicalNote: 'La confianza y las probabilidades son reportadas directamente por la API.',
    icon: FaBalanceScale,
    color: '#1e40af'
  },
  {
    title: 'Generación del informe',
    subtitle: 'Resultados y métricas',
    description: 'Se entregan la clase estimada, la confianza y la distribución de probabilidades en formato visual o exportable.',
    technicalNote: 'El informe puede descargarse o visualizarse en la interfaz.',
    icon: FaClipboardList,
    color: '#1e3a8a'
  }
];

function PipelineCard({ step, isActive }) {
  const IconComponent = step.icon;
  
  return (
    <motion.article
      className={`pipeline-card ${isActive ? 'active' : ''}`}
      initial={{ opacity: 0, x: -30 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: 30 }}
      transition={{ duration: 0.3, ease: 'easeOut' }}
    >
      <IconComponent size={26} style={{ color: step.color }} />
      <div className="pipeline-card__body">
        <span className="pipeline-card__subtitle">{step.subtitle}</span>
        <h4>{step.title}</h4>
        <p>{step.description}</p>
      </div>
      <div className="pipeline-card__footer">
        <span className="pipeline-card__tech">{step.technicalNote}</span>
        <div className="pipeline-card__progress">
          <motion.div 
            className="progress-fill"
            initial={{ width: 0 }}
            animate={{ width: isActive ? '100%' : 0 }}
            transition={{ duration: 0.3 }}
          />
        </div>
      </div>
    </motion.article>
  );
}

export function PipelineSection() {
  const sectionRef = useRef(null);
  const touchStartXRef = useRef(null);
  const isInView = useInView(sectionRef, { once: true, margin: '-100px' });
  const [activeIndex, setActiveIndex] = useState(0);

  const nextSlide = () => {
    setActiveIndex((prev) => (prev + 1) % pipelineStepsData.length);
  };

const prevSlide = () => {
    setActiveIndex((prev) => (prev - 1 + pipelineStepsData.length) % pipelineStepsData.length);
  };

  const handleTouchStart = (event) => {
    touchStartXRef.current = event.changedTouches[0]?.clientX ?? null;
  };

  const handleTouchEnd = (event) => {
    const endX = event.changedTouches[0]?.clientX;
    const startX = touchStartXRef.current;

    if (startX === null || typeof endX !== 'number') {
      touchStartXRef.current = null;
      return;
    }

    const deltaX = endX - startX;

    if (Math.abs(deltaX) < 40) {
      touchStartXRef.current = null;
      return;
    }

    if (deltaX < 0) {
      nextSlide();
    } else {
      prevSlide();
    }

    touchStartXRef.current = null;
  };

  return (
    <section className="home-section pipeline-section" id="process" ref={sectionRef}>
      <div className="pipeline-main">
        <div className="section-heading">
          <p className="eyebrow">Proceso técnico</p>
          <h3>¿Cómo transforma ITEL una muestra de voz en una predicción TEL?</h3>
          <p>Un pipeline de 5 etapas optimizado para detectar patrones acústicos asociados al Trastorno Específico del Lenguaje.</p>
        </div>

        <div className="pipeline-carousel">
          <button 
            className="carousel-nav carousel-nav--prev" 
            onClick={prevSlide}
            aria-label="Paso anterior"
          >
            <FaChevronLeft size={20} />
          </button>
          
          <div className="pipeline-carousel__track" onTouchStart={handleTouchStart} onTouchEnd={handleTouchEnd}>
            <AnimatePresence mode="wait">
              <PipelineCard
                key={activeIndex}
                step={pipelineStepsData[activeIndex]}
                isActive={true}
              />
            </AnimatePresence>
            
            <div className="pipeline-carousel__indicators">
              {pipelineStepsData.map((step, idx) => (
                <button
                  key={step.title}
                  className={`carousel-dot ${activeIndex === idx ? 'active' : ''}`}
                  onClick={() => setActiveIndex(idx)}
                  aria-label={`Ver ${step.title}`}
                />
              ))}
            </div>
          </div>
          
          <button 
            className="carousel-nav carousel-nav--next" 
            onClick={nextSlide}
            aria-label="Siguiente paso"
          >
            <FaChevronRight size={20} />
          </button>
        </div>
      </div>

      <motion.div 
        className="pipeline-metrics"
        initial={{ opacity: 0, y: 20 }}
        animate={isInView ? { opacity: 1, y: 0 } : {}}
        transition={{ delay: 0.3 }}
      >
        <div className="pipeline-metrics__item">
          <span className="pipeline-metrics__value">WAV</span>
          <span className="pipeline-metrics__label">Uno o más archivos</span>
        </div>
        <div className="pipeline-metrics__divider" />
        <div className="pipeline-metrics__item">
          <span className="pipeline-metrics__value">Sujeto</span>
          <span className="pipeline-metrics__label">Nivel de análisis</span>
        </div>
        <div className="pipeline-metrics__divider" />
        <div className="pipeline-metrics__item">
          <span className="pipeline-metrics__value">CZ · EN</span>
          <span className="pipeline-metrics__label">Idiomas disponibles</span>
        </div>
      </motion.div>

      <motion.article 
        className="pipeline-ethical"
        initial={{ opacity: 0, y: 20 }}
        animate={isInView ? { opacity: 1, y: 0 } : {}}
        transition={{ delay: 0.5 }}
      >
        <div className="pipeline-ethical__header">
          <p className="eyebrow">Uso responsable</p>
          <h3>Herramienta de apoyo, no diagnóstico autónomo</h3>
        </div>
        <p>La predicción debe ser interpretada por un profesional de la salud, integrando el contexto clínico y pruebas adicionales. ITEL ofrece una estimación basada en patrones acústicos, no reemplaza la evaluación neuropsicológica.</p>
      </motion.article>
    </section>
  );
}
