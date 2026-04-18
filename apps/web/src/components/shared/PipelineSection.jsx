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
    description: 'Se verifica que el archivo sea WAV, muestreo mínimo 16kHz, duración entre 2 y 10 segundos, y nivel de ruido aceptable.',
    technicalNote: 'Si falla, se rechaza con mensaje claro para el usuario.',
    icon: FaMicrophone,
    color: '#3b82f6'
  },
  {
    title: 'Preprocesamiento',
    subtitle: 'Limpieza y normalización',
    description: 'Remuestreo a 16kHz, eliminación de silencios iniciales/finales, normalización de amplitud y reducción de ruido de fondo.',
    technicalNote: 'Se aplica filtro pasa-altos y VAD (Voice Activity Detection).',
    icon: FaMagic,
    color: '#2563eb'
  },
  {
    title: 'Extracción de embeddings',
    subtitle: 'Representación acústicas',
    description: 'El audio procesado se transforma en un vector de características acústicas mediante un modelo preentrenado (Wav2Vec2 2.0).',
    technicalNote: 'Embedding de 768 dimensiones que captura rasgos fonéticos y prosódicos.',
    icon: FaNetworkWired,
    color: '#1d4ed8'
  },
  {
    title: 'Clasificación con confianza',
    subtitle: 'Predicción TEL',
    description: 'Un clasificador LightGBM evalúa el embedding y produce una predicción: "TEL positivo" o "TEL negativo", junto con un nivel de confianza (0-100%).',
    technicalNote: 'El umbral de confianza mínimo configurable es 70% para emitir resultado.',
    icon: FaBalanceScale,
    color: '#1e40af'
  },
  {
    title: 'Generación del informe',
    subtitle: 'Resultados y métricas',
    description: 'Se entregan: clase estimada, confianza, tiempos de cada etapa (ms) y metadatos del audio. Todo en formato JSON o visual.',
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
          <span className="pipeline-metrics__value">~1.2s</span>
          <span className="pipeline-metrics__label">Tiempo promedio</span>
        </div>
        <div className="pipeline-metrics__divider" />
        <div className="pipeline-metrics__item">
          <span className="pipeline-metrics__value">70%</span>
          <span className="pipeline-metrics__label">Confianza mínima</span>
        </div>
        <div className="pipeline-metrics__divider" />
        <div className="pipeline-metrics__item">
          <span className="pipeline-metrics__value">WAV</span>
          <span className="pipeline-metrics__label">16kHz</span>
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
        <div className="pipeline-ethical__stats">
          <span>Dataset clínico: 800+ niños</span>
          <span>Sensibilidad: 86% • Especificidad: 89%</span>
        </div>
      </motion.article>
    </section>
  );
}
