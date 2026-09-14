/**
 * VoxShield SOC Dashboard
 * Voice Threat Intelligence — Local Inference Only
 */

import { useState, useEffect, useRef, useCallback } from 'react'
import { api } from './api.js'

// ──────────────────────────────────────────────────────────────────────────────
// Helpers / constants
// ──────────────────────────────────────────────────────────────────────────────

const THREAT_COLORS = {
  LOW:      '#10b981',
  MEDIUM:   '#f59e0b',
  HIGH:     '#f97316',
  CRITICAL: '#ef4444',
  UNKNOWN:  '#64748b',
}

const THREAT_BG = {
  LOW:      'rgba(16,185,129,0.12)',
  MEDIUM:   'rgba(245,158,11,0.12)',
  HIGH:     'rgba(249,115,22,0.12)',
  CRITICAL: 'rgba(239,68,68,0.12)',
  UNKNOWN:  'rgba(100,116,139,0.12)',
}

function fmt(v, d = 4) {
  if (v === null || v === undefined) return '—'
  return typeof v === 'number' ? v.toFixed(d) : String(v)
}

function pct(v) {
  if (v === null || v === undefined) return 'N/A'
  return (v * 100).toFixed(2) + '%'
}

function probabilityPct(v) {
  if (v === null || v === undefined) return 'N/A'
  if (v <= 0) return '<0.01%'
  if (v >= 1) return '>99.99%'
  return (v * 100).toFixed(2) + '%'
}

function shortTs(ts) {
  if (!ts) return '—'
  try {
    const d = new Date(ts)
    return d.toLocaleTimeString('en-US', { hour12: false })
  } catch { return ts }
}

// ──────────────────────────────────────────────────────────────────────────────
// Sub-components
// ──────────────────────────────────────────────────────────────────────────────

function StatusDot({ ok, pulse = false }) {
  return (
    <span
      className={pulse ? 'pulse' : ''}
      style={{
        display: 'inline-block',
        width: 8, height: 8,
        borderRadius: '50%',
        background: ok ? 'var(--green)' : 'var(--red)',
        flexShrink: 0,
      }}
    />
  )
}

function ProbBar({ value, color }) {
  const p = Math.max(0, Math.min(1, value ?? 0)) * 100
  return (
    <div className="progress-bar" style={{ flex: 1 }}>
      <div
        className="progress-bar-fill"
        style={{ width: `${p}%`, background: color }}
      />
    </div>
  )
}

function ThreatBadge({ level }) {
  const color = THREAT_COLORS[level] || THREAT_COLORS.UNKNOWN
  const bg    = THREAT_BG[level]    || THREAT_BG.UNKNOWN
  return (
    <span style={{
      background: bg,
      color,
      border: `1px solid ${color}40`,
      padding: '3px 10px',
      borderRadius: 4,
      fontSize: 11,
      fontWeight: 700,
      letterSpacing: '0.06em',
      textTransform: 'uppercase',
      lineHeight: 1,
      display: 'inline-flex',
      alignItems: 'center',
    }}>
      {level || 'UNKNOWN'}
    </span>
  )
}

/**
 * MetricRow — a label/value pair used inside detail cards.
 * label: 12px muted
 * value: 13px primary (or secondary if dim=true)
 */
function MetricRow({ label, value, mono = true, dim = false }) {
  return (
    <div style={{
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      padding: '5px 0',
      lineHeight: 'var(--lh-compact)',
    }}>
      <span style={{ color: 'var(--text-muted)', fontSize: 'var(--fs-xs)' }}>{label}</span>
      <span style={{
        color: dim ? 'var(--text-secondary)' : 'var(--text-primary)',
        fontFamily: mono ? 'var(--font-mono)' : undefined,
        fontSize: 'var(--fs-sm)',
      }}>
        {value ?? '—'}
      </span>
    </div>
  )
}

/**
 * SectionTitle — uppercase label inside a card, above a section of content.
 * 11px, 600 weight, tracked — intentionally small as a section divider label,
 * not a heading.
 */
function SectionTitle({ children, icon }) {
  return (
    <div style={{
      fontSize: 'var(--fs-xxs)',    /* 11px */
      fontWeight: 600,
      letterSpacing: '0.08em',
      textTransform: 'uppercase',
      color: 'var(--text-muted)',
      marginBottom: 12,
      display: 'flex',
      alignItems: 'center',
      gap: 6,
    }}>
      {icon && <span style={{ fontSize: 14 }}>{icon}</span>}
      {children}
    </div>
  )
}

// ──────────────────────────────────────────────────────────────────────────────
// Header
// ──────────────────────────────────────────────────────────────────────────────

function Header({ health, modelReady }) {
  const [now, setNow] = useState(new Date())
  useEffect(() => {
    const t = setInterval(() => setNow(new Date()), 1000)
    return () => clearInterval(t)
  }, [])

  return (
    <header style={{
      background: 'var(--bg-secondary)',
      borderBottom: '1px solid var(--border)',
      padding: '0 24px',
      height: 60,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      flexShrink: 0,
    }}>
      {/* Logo */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <div style={{
            width: 34, height: 34,
            background: 'linear-gradient(135deg, #1e40af, #2563eb)',
            borderRadius: 8,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: 18,
            boxShadow: '0 0 12px rgba(37,99,235,0.4)',
            flexShrink: 0,
          }}>🛡</div>
          <div>
            {/* Brand name — 18px, 700 */}
            <div style={{
              fontWeight: 700,
              fontSize: 18,
              letterSpacing: '0.06em',
              color: 'var(--text-primary)',
              lineHeight: 1.2,
            }}>
              VOXSHIELD
            </div>
            {/* Tagline — 10px, intentional micro-label */}
            <div style={{
              fontSize: 10,
              color: 'var(--text-muted)',
              letterSpacing: '0.12em',
              textTransform: 'uppercase',
              lineHeight: 1.3,
              marginTop: 1,
            }}>
              VOICE THREAT INTELLIGENCE
            </div>
          </div>
        </div>

        <div style={{ width: 1, height: 28, background: 'var(--border)', flexShrink: 0 }} />

        <span style={{
          fontSize: 'var(--fs-xxs)',
          fontWeight: 600,
          letterSpacing: '0.08em',
          padding: '4px 10px',
          background: 'rgba(16,185,129,0.1)',
          color: 'var(--green)',
          border: '1px solid rgba(16,185,129,0.3)',
          borderRadius: 4,
        }}>
          ● LOCAL INFERENCE
        </span>
      </div>

      {/* Right side */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 24 }}>
        {/* Model status */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
          <StatusDot ok={modelReady} pulse={!modelReady} />
          <span style={{ fontSize: 'var(--fs-xs)', color: 'var(--text-secondary)' }}>
            {modelReady ? 'Model Ready' : 'Model Offline'}
          </span>
        </div>

        {/* API status */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
          <StatusDot ok={health !== null} pulse={health === null} />
          <span style={{ fontSize: 'var(--fs-xs)', color: 'var(--text-secondary)' }}>
            {health !== null ? 'API Connected' : 'API Offline'}
          </span>
        </div>

        {/* Clock */}
        <div style={{
          fontFamily: 'var(--font-mono)',
          fontSize: 'var(--fs-xs)',
          color: 'var(--text-muted)',
          minWidth: 72,
          letterSpacing: '0.04em',
        }}>
          {now.toLocaleTimeString('en-US', { hour12: false })}
        </div>
      </div>
    </header>
  )
}

// ──────────────────────────────────────────────────────────────────────────────
// Dashboard Hero — main page title
// ──────────────────────────────────────────────────────────────────────────────

function DashboardHero({ modelReady }) {
  return (
    <div style={{
      padding: '20px 24px 4px',
      maxWidth: 1400,
      width: '100%',
      margin: '0 auto',
    }}>
      {/* Main heading — 30px, 700 */}
      <h1 style={{
        fontSize: 'var(--fs-hero)',
        fontWeight: 700,
        lineHeight: 'var(--lh-heading)',
        color: 'var(--text-primary)',
        letterSpacing: '-0.01em',
        margin: 0,
      }}>
        Stop Voice Fraud Before It Happens
      </h1>
      {/* Subtitle — 14px body */}
      <p style={{
        fontSize: 'var(--fs-body)',
        color: 'var(--text-secondary)',
        marginTop: 6,
        lineHeight: 'var(--lh-body)',
      }}>
        Real-time spoof detection powered by VoxShieldNet — 100% local, no external APIs.
        {!modelReady && (
          <span style={{ marginLeft: 10, color: 'var(--yellow)', fontWeight: 500 }}>
            ⚠ Model offline — training required.
          </span>
        )}
      </p>
    </div>
  )
}

// ──────────────────────────────────────────────────────────────────────────────
// Audio Upload + Analysis Panel
// ──────────────────────────────────────────────────────────────────────────────

function UploadPanel({ onResult, modelReady }) {
  const [file, setFile]         = useState(null)
  const [dragging, setDragging] = useState(false)
  const [loading, setLoading]   = useState(false)
  const [error, setError]       = useState(null)
  const inputRef = useRef(null)

  const ACCEPTED     = ['audio/flac', 'audio/wav', 'audio/ogg', 'audio/mpeg', 'audio/mp4', 'audio/x-wav']
  const ACCEPTED_EXT = ['.flac', '.wav', '.ogg', '.mp3', '.m4a']

  function handleFile(f) {
    if (!f) return
    setError(null)
    setFile(f)
  }

  function onDrop(e) {
    e.preventDefault(); setDragging(false)
    const f = e.dataTransfer.files[0]
    if (f) handleFile(f)
  }

  async function analyze() {
    if (!file) return
    setLoading(true); setError(null)
    try {
      const result = await api.predict(file)
      onResult(result)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const sizeKB = file ? (file.size / 1024).toFixed(1) : null

  return (
    <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
      <SectionTitle icon="🎙">Analyze Voice</SectionTitle>

      {/* Drop zone */}
      <div
        onClick={() => inputRef.current?.click()}
        onDragOver={e => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        style={{
          border: `2px dashed ${dragging ? 'var(--accent)' : file ? 'var(--green)' : 'var(--border)'}`,
          borderRadius: 'var(--radius)',
          padding: '28px 16px',
          textAlign: 'center',
          cursor: 'pointer',
          background: dragging ? 'var(--accent-glow)' : file ? 'var(--green-dim)' : 'transparent',
          transition: 'var(--transition)',
        }}
      >
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED_EXT.join(',')}
          style={{ display: 'none' }}
          onChange={e => handleFile(e.target.files[0])}
        />
        {file ? (
          <div>
            <div style={{ fontSize: 24, marginBottom: 6 }}>🎵</div>
            <div style={{
              color: 'var(--text-primary)',
              fontWeight: 600,
              fontSize: 'var(--fs-body)',
              lineHeight: 'var(--lh-compact)',
            }}>
              {file.name}
            </div>
            <div style={{
              color: 'var(--text-muted)',
              fontSize: 'var(--fs-xs)',
              marginTop: 3,
            }}>
              {sizeKB} KB · {file.type || 'audio'}
            </div>
          </div>
        ) : (
          <div>
            <div style={{ fontSize: 24, marginBottom: 10 }}>⬆</div>
            <div style={{
              color: 'var(--text-secondary)',
              fontSize: 'var(--fs-body)',
              fontWeight: 500,
            }}>
              Drop audio file here or click to upload
            </div>
            <div style={{
              color: 'var(--text-muted)',
              fontSize: 'var(--fs-xs)',
              marginTop: 5,
            }}>
              {ACCEPTED_EXT.join(' · ')} · max 25 MB
            </div>
          </div>
        )}
      </div>

      {/* Error */}
      {error && (
        <div style={{
          background: 'var(--red-dim)',
          border: '1px solid rgba(239,68,68,0.3)',
          color: 'var(--red)',
          borderRadius: 'var(--radius-sm)',
          padding: '9px 12px',
          fontSize: 'var(--fs-sm)',
          lineHeight: 'var(--lh-compact)',
        }}>
          ⚠ {error}
        </div>
      )}

      {/* Analyze button */}
      <button
        className="btn btn-primary"
        style={{ width: '100%', padding: '11px' }}
        disabled={!file || loading || !modelReady}
        onClick={analyze}
        title={!modelReady ? 'Model offline — train first' : undefined}
      >
        {loading
          ? <><span className="spinner" /> Analyzing...</>
          : !modelReady
          ? '⚠ Model Offline — Training Required'
          : '⚡ Analyze Audio'
        }
      </button>

      {!modelReady && (
        <div style={{
          textAlign: 'center',
          color: 'var(--yellow)',
          fontSize: 'var(--fs-xs)',
          padding: '5px 8px',
          background: 'var(--yellow-dim)',
          borderRadius: 'var(--radius-sm)',
          fontWeight: 500,
        }}>
          MODEL OFFLINE — Run training to enable analysis
        </div>
      )}
    </div>
  )
}

// ──────────────────────────────────────────────────────────────────────────────
// Result Panel
// ──────────────────────────────────────────────────────────────────────────────

function ResultPanel({ result }) {
  if (!result) {
    return (
      <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: 10, minHeight: 200 }}>
        <SectionTitle icon="📊">Analysis Result</SectionTitle>
        <div style={{
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          color: 'var(--text-muted)',
          fontSize: 'var(--fs-sm)',
          padding: '28px 0',
          gap: 10,
        }}>
          <span style={{ fontSize: 36 }}>🔍</span>
          <span>Upload and analyze audio to see results</span>
        </div>
      </div>
    )
  }

  const { classification, status } = result

  if (status === 'unavailable') {
    return (
      <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
        <SectionTitle icon="⚠">Analysis Result</SectionTitle>
        <div style={{
          background: 'var(--yellow-dim)',
          border: '1px solid rgba(245,158,11,0.3)',
          borderRadius: 'var(--radius)',
          padding: 20,
          textAlign: 'center',
        }}>
          <div style={{ fontSize: 32, marginBottom: 10 }}>🔌</div>
          <div style={{
            color: 'var(--yellow)',
            fontWeight: 700,
            fontSize: 'var(--fs-body)',
          }}>
            MODEL OFFLINE
          </div>
          <div style={{
            color: 'var(--text-secondary)',
            fontSize: 'var(--fs-sm)',
            marginTop: 6,
            lineHeight: 'var(--lh-body)',
          }}>
            {result.recommended_action}
          </div>
        </div>
      </div>
    )
  }

  if (status === 'error') {
    return (
      <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
        <SectionTitle icon="✕">Analysis Error</SectionTitle>
        <div style={{
          background: 'var(--red-dim)',
          border: '1px solid rgba(239,68,68,0.3)',
          borderRadius: 'var(--radius)',
          padding: 16,
        }}>
          <div style={{
            color: 'var(--red)',
            fontWeight: 600,
            fontSize: 'var(--fs-sm)',
            lineHeight: 'var(--lh-compact)',
          }}>
            ⚠ {result.error}
          </div>
        </div>
      </div>
    )
  }

  const isSpoof   = classification === 'SPOOF'
  const mainColor = isSpoof ? 'var(--red)' : 'var(--green)'
  const mainBg    = isSpoof ? 'var(--red-dim)' : 'var(--green-dim)'
  const spoofP    = result.spoof_probability ?? 0
  const bfP       = result.bona_fide_probability ?? 0

  return (
    <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
      <SectionTitle icon="📊">Analysis Result</SectionTitle>

      {/* Main verdict — dominant typography */}
      <div style={{
        background: mainBg,
        border: `1px solid ${mainColor}40`,
        borderRadius: 'var(--radius)',
        padding: '20px 16px',
        textAlign: 'center',
        boxShadow: isSpoof ? '0 0 20px var(--red-glow)' : 'none',
      }}>
        <div style={{ fontSize: 30, marginBottom: 6 }}>
          {isSpoof ? '🚨' : '✅'}
        </div>
        {/* Verdict — 24px, 800 — most prominent element */}
        <div style={{
          fontSize: 24,
          fontWeight: 800,
          color: mainColor,
          letterSpacing: '0.04em',
          lineHeight: 1.2,
        }}>
          {isSpoof ? 'SYNTHETIC SPEECH' : 'BONA FIDE'}
        </div>
        <div style={{
          fontSize: 'var(--fs-sm)',
          color: 'var(--text-secondary)',
          marginTop: 5,
        }}>
          {isSpoof ? 'Spoof / AI-generated audio detected' : 'Genuine human speech'}
        </div>
      </div>

      {/* Probabilities */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        {/* Spoof probability */}
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
            <span style={{ fontSize: 'var(--fs-xs)', color: 'var(--text-muted)' }}>Spoof Probability</span>
            <span style={{
              fontSize: 'var(--fs-sm)',
              fontFamily: 'var(--font-mono)',
              color: 'var(--red)',
              fontWeight: 600,
            }}>
              {probabilityPct(spoofP)}
            </span>
          </div>
          <ProbBar value={spoofP} color="var(--red)" />
        </div>

        {/* Bona-fide probability */}
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
            <span style={{ fontSize: 'var(--fs-xs)', color: 'var(--text-muted)' }}>Bona-Fide Probability</span>
            <span style={{
              fontSize: 'var(--fs-sm)',
              fontFamily: 'var(--font-mono)',
              color: 'var(--green)',
              fontWeight: 600,
            }}>
              {probabilityPct(bfP)}
            </span>
          </div>
          <ProbBar value={bfP} color="var(--green)" />
        </div>
      </div>

      {/* Metrics grid */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
        {[
          { label: 'Confidence',  value: probabilityPct(result.confidence) },
          { label: 'Risk Score',  value: fmt(result.risk_score, 3) },
          { label: 'Threshold',   value: fmt(result.decision_threshold, 4) },
          { label: 'Latency',     value: `${fmt(result.latency_ms, 1)} ms` },
        ].map(({ label, value }) => (
          <div key={label} style={{
            background: 'var(--bg-secondary)',
            border: '1px solid var(--border)',
            borderRadius: 'var(--radius-sm)',
            padding: '10px 12px',
          }}>
            <div style={{
              fontSize: 'var(--fs-xxs)',
              color: 'var(--text-muted)',
              marginBottom: 4,
              textTransform: 'uppercase',
              letterSpacing: '0.04em',
            }}>
              {label}
            </div>
            <div style={{
              fontSize: 'var(--fs-sm)',
              fontFamily: 'var(--font-mono)',
              color: 'var(--text-primary)',
              fontWeight: 500,
            }}>
              {value}
            </div>
          </div>
        ))}
      </div>

      {/* Threat level */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '8px 0',
      }}>
        <span style={{ fontSize: 'var(--fs-sm)', color: 'var(--text-secondary)', fontWeight: 500 }}>
          Threat Level
        </span>
        <ThreatBadge level={result.threat_level} />
      </div>

      {/* Recommended action */}
      <div style={{
        background: 'var(--bg-secondary)',
        border: '1px solid var(--border)',
        borderRadius: 'var(--radius-sm)',
        padding: '10px 12px',
      }}>
        <div style={{
          fontSize: 'var(--fs-xxs)',
          color: 'var(--text-muted)',
          marginBottom: 5,
          textTransform: 'uppercase',
          letterSpacing: '0.06em',
          fontWeight: 600,
        }}>
          Recommended Action
        </div>
        <div style={{
          fontSize: 'var(--fs-sm)',
          color: 'var(--text-secondary)',
          lineHeight: 'var(--lh-body)',
        }}>
          {result.recommended_action}
        </div>
      </div>

      {/* Footer meta */}
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        fontSize: 'var(--fs-xxs)',
        color: 'var(--text-muted)',
      }}>
        <span>Device: {result.device || '—'}</span>
        <span>Mode: {result.inference_mode || '—'}</span>
        <span>v{result.model_version || '—'}</span>
      </div>
    </div>
  )
}

// ──────────────────────────────────────────────────────────────────────────────
// Model Info Panel
// ──────────────────────────────────────────────────────────────────────────────

function ModelPanel({ modelInfo, onReload }) {
  const [reloading, setReloading] = useState(false)

  async function handleReload() {
    setReloading(true)
    try { await api.reloadModel(); onReload() }
    catch (e) { console.error(e) }
    finally { setReloading(false) }
  }

  const ready  = modelInfo?.status === 'ready'
  const vm     = modelInfo?.val_metrics || {}
  const dsType = modelInfo?.dataset_type || '—'
  const isReal = dsType === 'ASVspoof5'

  return (
    <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      {/* Card header row */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <SectionTitle icon="🧠">Model</SectionTitle>
        <button
          className="btn btn-secondary"
          style={{ padding: '5px 10px', fontSize: 'var(--fs-xxs)' }}
          onClick={handleReload}
          disabled={reloading}
        >
          {reloading ? '...' : '↺ Reload'}
        </button>
      </div>

      {!ready ? (
        <div style={{
          background: 'var(--yellow-dim)',
          border: '1px solid rgba(245,158,11,0.3)',
          borderRadius: 'var(--radius-sm)',
          padding: '14px',
          textAlign: 'center',
        }}>
          <div style={{
            color: 'var(--yellow)',
            fontWeight: 700,
            marginBottom: 5,
            fontSize: 'var(--fs-body)',
          }}>
            ⚠ MODEL OFFLINE
          </div>
          <div style={{ color: 'var(--text-muted)', fontSize: 'var(--fs-sm)' }}>
            Training required
          </div>
          <div style={{
            color: 'var(--text-muted)',
            fontSize: 'var(--fs-xxs)',
            marginTop: 6,
            fontFamily: 'var(--font-mono)',
            lineHeight: 'var(--lh-compact)',
          }}>
            python -m training.train --asvspoof5-dir &lt;PATH&gt;
          </div>
        </div>
      ) : (
        <>
          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
            {isReal
              ? <span className="badge badge-blue">ASVspoof5</span>
              : <span className="badge badge-gray">demo</span>
            }
            <span className="badge badge-green">LOCAL ONLY</span>
            <span className="badge badge-gray">pretrained: NO</span>
          </div>

          {/* Model details */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
            {[
              ['Name',       modelInfo.model_name],
              ['Version',    modelInfo.model_version],
              ['Parameters', modelInfo.parameter_count?.toLocaleString()],
              ['Device',     modelInfo.device],
              ['Dataset',    dsType],
              ['Epoch',      modelInfo.epoch],
              ['Threshold',  fmt(modelInfo.decision_threshold, 6)],
            ].map(([k, v]) => (
              <MetricRow key={k} label={k} value={v} />
            ))}
          </div>

          {/* Validation metrics */}
          {Object.keys(vm).length > 0 && (
            <>
              <div className="divider" />
              <div style={{
                fontSize: 'var(--fs-xxs)',
                color: 'var(--text-muted)',
                marginBottom: 6,
                letterSpacing: '0.06em',
                textTransform: 'uppercase',
                fontWeight: 600,
              }}>
                Validation Metrics
              </div>
              <div style={{
                background: 'var(--bg-secondary)',
                borderRadius: 'var(--radius-sm)',
                padding: '10px 12px',
                display: 'grid',
                gridTemplateColumns: '1fr 1fr',
                gap: '5px 16px',
              }}>
                {[
                  ['ROC-AUC', vm.roc_auc],
                  ['EER',     vm.eer],
                  ['F1',      vm.f1],
                  ['Recall',  vm.recall],
                  ['FPR',     vm.fpr],
                  ['FNR',     vm.fnr],
                ].filter(([,v]) => v !== undefined).map(([k, v]) => (
                  <div key={k} style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                  }}>
                    <span style={{ color: 'var(--text-muted)', fontSize: 'var(--fs-xs)' }}>{k}</span>
                    <span style={{
                      fontFamily: 'var(--font-mono)',
                      fontSize: 'var(--fs-sm)',
                      color: 'var(--text-secondary)',
                    }}>
                      {fmt(v, 4)}
                    </span>
                  </div>
                ))}
              </div>
              <div style={{
                fontSize: 'var(--fs-xxs)',
                color: 'var(--text-muted)',
                lineHeight: 'var(--lh-body)',
              }}>
                ⚠ Metrics from validation split only. Not production guarantees.
              </div>
            </>
          )}
        </>
      )}
    </div>
  )
}

// ──────────────────────────────────────────────────────────────────────────────
// Timeline Panel
// ──────────────────────────────────────────────────────────────────────────────

function TimelinePanel({ timeline }) {
  if (!timeline || timeline.length === 0) {
    return (
      <div className="card">
        <SectionTitle icon="📡">Live Timeline</SectionTitle>
        <div style={{
          color: 'var(--text-muted)',
          fontSize: 'var(--fs-sm)',
          padding: '14px 0',
          textAlign: 'center',
        }}>
          No analysis events yet
        </div>
      </div>
    )
  }

  return (
    <div className="card">
      <SectionTitle icon="📡">Live Timeline</SectionTitle>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
        {timeline.slice(0, 12).map((ev) => {
          const isSpoof = ev.classification === 'SPOOF'
          const color   = THREAT_COLORS[ev.threat_level] || 'var(--text-muted)'
          return (
            <div key={ev.id} style={{
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              padding: '6px 10px',
              borderRadius: 'var(--radius-sm)',
              background: 'var(--bg-secondary)',
            }}>
              {/* Timestamp */}
              <span style={{
                color: 'var(--text-muted)',
                fontFamily: 'var(--font-mono)',
                fontSize: 'var(--fs-xs)',
                minWidth: 68,
                flexShrink: 0,
              }}>
                {shortTs(ev.timestamp)}
              </span>

              {/* Classification */}
              <span style={{
                color: isSpoof ? 'var(--red)' : 'var(--green)',
                fontWeight: 700,
                fontSize: 'var(--fs-xxs)',
                minWidth: 72,
                flexShrink: 0,
                letterSpacing: '0.03em',
              }}>
                {ev.classification}
              </span>

              {/* Prob bar */}
              <div style={{ flex: 1 }}>
                <div style={{
                  height: 3,
                  background: 'var(--border)',
                  borderRadius: 2,
                  overflow: 'hidden',
                }}>
                  <div style={{
                    height: '100%',
                    width: `${(ev.spoof_prob ?? 0) * 100}%`,
                    background: color,
                    borderRadius: 2,
                  }} />
                </div>
              </div>

              {/* Probability */}
              <span style={{
                color,
                fontFamily: 'var(--font-mono)',
                fontSize: 'var(--fs-xs)',
                minWidth: 52,
                textAlign: 'right',
                flexShrink: 0,
              }}>
                {ev.spoof_prob !== null && ev.spoof_prob !== undefined
                  ? probabilityPct(ev.spoof_prob)
                  : '—'
                }
              </span>

              <ThreatBadge level={ev.threat_level} />
            </div>
          )
        })}
      </div>
    </div>
  )
}

// ──────────────────────────────────────────────────────────────────────────────
// Incidents Table
// ──────────────────────────────────────────────────────────────────────────────

function IncidentsPanel({ incidents }) {
  if (!incidents || incidents.length === 0) {
    return (
      <div className="card">
        <SectionTitle icon="🗂">Incident History</SectionTitle>
        <div style={{
          color: 'var(--text-muted)',
          fontSize: 'var(--fs-sm)',
          padding: '14px 0',
          textAlign: 'center',
        }}>
          No incidents logged yet
        </div>
      </div>
    )
  }

  return (
    <div className="card">
      <SectionTitle icon="🗂">Incident History</SectionTitle>
      <div style={{ overflowX: 'auto' }}>
        <table className="data-table">
          <thead>
            <tr>
              {['Time', 'Classification', 'Spoof P', 'Risk', 'Threat', 'Latency', 'File'].map(h => (
                <th key={h}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {incidents.slice(0, 20).map((inc) => (
              <tr key={inc.id}>
                <td style={{
                  fontFamily: 'var(--font-mono)',
                  color: 'var(--text-muted)',
                  whiteSpace: 'nowrap',
                  fontSize: 'var(--fs-xs)',
                }}>
                  {shortTs(inc.timestamp)}
                </td>
                <td>
                  <span style={{
                    color: inc.classification === 'SPOOF' ? 'var(--red)' : 'var(--green)',
                    fontWeight: 700,
                    fontSize: 'var(--fs-sm)',
                    letterSpacing: '0.03em',
                  }}>
                    {inc.classification}
                  </span>
                </td>
                <td style={{
                  fontFamily: 'var(--font-mono)',
                  color: 'var(--text-secondary)',
                  fontSize: 'var(--fs-sm)',
                }}>
                  {inc.spoof_prob !== null && inc.spoof_prob !== undefined
                    ? probabilityPct(inc.spoof_prob)
                    : '—'
                  }
                </td>
                <td style={{
                  fontFamily: 'var(--font-mono)',
                  color: 'var(--text-secondary)',
                  fontSize: 'var(--fs-sm)',
                }}>
                  {inc.risk_score !== null && inc.risk_score !== undefined
                    ? inc.risk_score.toFixed(3)
                    : '—'
                  }
                </td>
                <td>
                  <ThreatBadge level={inc.threat_level} />
                </td>
                <td style={{
                  fontFamily: 'var(--font-mono)',
                  color: 'var(--text-muted)',
                  fontSize: 'var(--fs-xs)',
                  whiteSpace: 'nowrap',
                }}>
                  {inc.latency_ms !== null ? `${inc.latency_ms.toFixed(0)}ms` : '—'}
                </td>
                <td style={{
                  color: 'var(--text-muted)',
                  maxWidth: 120,
                  fontSize: 'var(--fs-sm)',
                }} className="truncate">
                  {inc.filename || '—'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

// ──────────────────────────────────────────────────────────────────────────────
// Stats Panel (Threat Distribution)
// ──────────────────────────────────────────────────────────────────────────────

function StatsPanel({ stats }) {
  if (!stats || stats.total === 0) {
    return (
      <div className="card">
        <SectionTitle icon="📈">Threat Distribution</SectionTitle>
        <div style={{
          color: 'var(--text-muted)',
          fontSize: 'var(--fs-sm)',
          textAlign: 'center',
          padding: '14px 0',
        }}>
          No data yet
        </div>
      </div>
    )
  }

  const byThr = stats.by_threat || {}

  return (
    <div className="card">
      <SectionTitle icon="📈">Threat Distribution</SectionTitle>

      {/* KPI grid — large prominent numbers */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8, marginBottom: 14 }}>
        {[
          { label: 'Total',       value: stats.total },
          { label: 'Spoof Rate',  value: stats.spoof_rate !== null ? pct(stats.spoof_rate) : '—' },
          { label: 'Avg Latency', value: stats.avg_latency_ms ? `${stats.avg_latency_ms}ms` : '—' },
        ].map(({ label, value }) => (
          <div key={label} style={{
            background: 'var(--bg-secondary)',
            border: '1px solid var(--border)',
            borderRadius: 'var(--radius-sm)',
            padding: '10px 8px',
            textAlign: 'center',
          }}>
            {/* Metric value — 28px, 700 — visually dominant */}
            <div style={{
              fontSize: 28,
              fontWeight: 700,
              fontFamily: 'var(--font-mono)',
              color: 'var(--text-primary)',
              lineHeight: 1.2,
            }}>
              {value}
            </div>
            {/* Label — 12px muted */}
            <div style={{
              fontSize: 'var(--fs-xs)',
              color: 'var(--text-muted)',
              marginTop: 4,
              fontWeight: 500,
            }}>
              {label}
            </div>
          </div>
        ))}
      </div>

      {/* Threat breakdown */}
      {Object.keys(byThr).length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          {['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'].filter(l => byThr[l]).map(level => (
            <div key={level} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ minWidth: 72, flexShrink: 0 }}>
                <ThreatBadge level={level} />
              </span>
              <div className="progress-bar" style={{ flex: 1 }}>
                <div className="progress-bar-fill" style={{
                  width: `${(byThr[level] / stats.total) * 100}%`,
                  background: THREAT_COLORS[level],
                }} />
              </div>
              <span style={{
                fontSize: 'var(--fs-sm)',
                fontFamily: 'var(--font-mono)',
                minWidth: 28,
                textAlign: 'right',
                color: 'var(--text-secondary)',
                flexShrink: 0,
              }}>
                {byThr[level]}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// ──────────────────────────────────────────────────────────────────────────────
// System Status
// ──────────────────────────────────────────────────────────────────────────────

function SystemStatus({ health, modelInfo }) {
  const ready  = modelInfo?.status === 'ready'
  const checks = [
    { label: 'API Server',       ok: health !== null, value: health ? 'Connected' : 'Offline' },
    { label: 'Model',            ok: ready,           value: ready ? 'Loaded' : 'Offline' },
    { label: 'Inference',        ok: ready,           value: health?.inference_mode || '—' },
    { label: 'Device',           ok: true,            value: health?.device || '—' },
    { label: 'External APIs',    ok: true,            value: 'NONE' },
    { label: 'Pretrained Model', ok: true,            value: 'No' },
  ]

  return (
    <div className="card">
      <SectionTitle icon="🖥">System Status</SectionTitle>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
        {checks.map(({ label, ok, value }) => (
          <div key={label} style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: '5px 0',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
              <StatusDot ok={ok} />
              <span style={{
                fontSize: 'var(--fs-sm)',
                color: 'var(--text-secondary)',
              }}>
                {label}
              </span>
            </div>
            <span style={{
              fontSize: 'var(--fs-sm)',
              fontFamily: 'var(--font-mono)',
              color: ok ? 'var(--text-secondary)' : 'var(--red)',
            }}>
              {value}
            </span>
          </div>
        ))}
      </div>

      <div className="divider" />

      <div style={{
        fontSize: 'var(--fs-xs)',
        color: 'var(--text-muted)',
        lineHeight: 'var(--lh-body)',
        background: 'var(--bg-secondary)',
        borderRadius: 'var(--radius-sm)',
        padding: '9px 10px',
      }}>
        All inference runs locally. No audio data leaves this machine.
        VoxShieldNet is trained from random initialization with no pretrained components.
      </div>
    </div>
  )
}

// ──────────────────────────────────────────────────────────────────────────────
// Main App
// ──────────────────────────────────────────────────────────────────────────────

export default function App() {
  const [health,    setHealth]    = useState(null)
  const [modelInfo, setModelInfo] = useState(null)
  const [result,    setResult]    = useState(null)
  const [timeline,  setTimeline]  = useState([])
  const [incidents, setIncidents] = useState([])
  const [stats,     setStats]     = useState(null)
  const [apiError,  setApiError]  = useState(false)

  const modelReady = modelInfo?.status === 'ready'

  // ── Polling ───────────────────────────────────────────────────────────────
  const refreshData = useCallback(async () => {
    try {
      const [h, mi] = await Promise.all([api.health(), api.modelInfo()])
      setHealth(h)
      setModelInfo(mi)
      setApiError(false)
    } catch (e) {
      setApiError(true)
    }
  }, [])

  const refreshIncidents = useCallback(async () => {
    try {
      const [tl, inc, st] = await Promise.all([
        api.timeline(20),
        api.incidents({ limit: 30 }),
        api.incidentStats(),
      ])
      setTimeline(tl.timeline || [])
      setIncidents(inc.incidents || [])
      setStats(st)
    } catch (_) {}
  }, [])

  useEffect(() => {
    refreshData()
    refreshIncidents()
    const t1 = setInterval(refreshData, 5000)
    const t2 = setInterval(refreshIncidents, 3000)
    return () => { clearInterval(t1); clearInterval(t2) }
  }, [refreshData, refreshIncidents])

  function handleResult(r) {
    setResult(r)
    setTimeout(refreshIncidents, 500)
  }

  // ── Layout ────────────────────────────────────────────────────────────────
  return (
    <div style={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
      <Header health={health} modelReady={modelReady} />

      {/* API offline banner */}
      {apiError && (
        <div style={{
          background: 'var(--red-dim)',
          borderBottom: '1px solid rgba(239,68,68,0.3)',
          padding: '9px 24px',
          color: 'var(--red)',
          fontSize: 'var(--fs-sm)',
          textAlign: 'center',
        }}>
          ⚠ Cannot connect to VoxShield API (localhost:8000) — Start the backend with:
          <code style={{
            marginLeft: 10,
            fontFamily: 'var(--font-mono)',
            fontSize: 'var(--fs-xs)',
          }}>
            uvicorn backend.app:app --port 8000
          </code>
        </div>
      )}

      {/* Page hero title */}
      <DashboardHero modelReady={modelReady} />

      {/* Main content grid */}
      <div style={{
        flex: 1,
        display: 'grid',
        gridTemplateColumns: '320px 1fr 280px',
        gridTemplateRows: 'auto 1fr',
        gap: 16,
        padding: '12px 16px 16px',
        maxWidth: 1400,
        width: '100%',
        margin: '0 auto',
        alignItems: 'start',
      }}>
        {/* LEFT COLUMN */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <UploadPanel onResult={handleResult} modelReady={modelReady} />
          <ResultPanel result={result} />
        </div>

        {/* MIDDLE COLUMN */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16, minWidth: 0 }}>
          <TimelinePanel timeline={timeline} />
          <IncidentsPanel incidents={incidents} />
        </div>

        {/* RIGHT COLUMN */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <ModelPanel modelInfo={modelInfo} onReload={refreshData} />
          <StatsPanel stats={stats} />
          <SystemStatus health={health} modelInfo={modelInfo} />
        </div>
      </div>

      {/* Footer */}
      <footer style={{
        borderTop: '1px solid var(--border)',
        padding: '9px 24px',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        fontSize: 'var(--fs-nano)',
        color: 'var(--text-muted)',
        flexWrap: 'wrap',
        gap: 4,
      }}>
        <span>VoxShield Independent v1.0 · Voice Threat Intelligence · Local Inference Only</span>
        <span>VoxShieldNet trained from random initialization · No pretrained components · No external APIs</span>
        <span>© 2026 · SIH Demo Build</span>
      </footer>
    </div>
  )
}
