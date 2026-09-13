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
  const pct = Math.max(0, Math.min(1, value ?? 0)) * 100
  return (
    <div className="progress-bar" style={{ flex: 1 }}>
      <div
        className="progress-bar-fill"
        style={{ width: `${pct}%`, background: color }}
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
      padding: '2px 10px',
      borderRadius: 4,
      fontSize: 11,
      fontWeight: 700,
      letterSpacing: '0.08em',
      textTransform: 'uppercase',
    }}>
      {level || 'UNKNOWN'}
    </span>
  )
}

function MetricRow({ label, value, mono = true, dim = false }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '5px 0' }}>
      <span style={{ color: 'var(--text-muted)', fontSize: 12 }}>{label}</span>
      <span style={{
        color: dim ? 'var(--text-secondary)' : 'var(--text-primary)',
        fontFamily: mono ? 'var(--font-mono)' : undefined,
        fontSize: 12,
      }}>
        {value ?? '—'}
      </span>
    </div>
  )
}

function SectionTitle({ children, icon }) {
  return (
    <div style={{
      fontSize: 10,
      fontWeight: 700,
      letterSpacing: '0.12em',
      textTransform: 'uppercase',
      color: 'var(--text-muted)',
      marginBottom: 10,
      display: 'flex',
      alignItems: 'center',
      gap: 6,
    }}>
      {icon && <span style={{ fontSize: 13 }}>{icon}</span>}
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
      height: 56,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      flexShrink: 0,
    }}>
      {/* Logo */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{
            width: 32, height: 32,
            background: 'linear-gradient(135deg, #1e40af, #2563eb)',
            borderRadius: 8,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: 16,
            boxShadow: '0 0 12px rgba(37,99,235,0.4)',
          }}>🛡</div>
          <div>
            <div style={{ fontWeight: 700, fontSize: 15, letterSpacing: '0.05em', color: 'var(--text-primary)' }}>
              VOXSHIELD
            </div>
            <div style={{ fontSize: 10, color: 'var(--text-muted)', letterSpacing: '0.1em' }}>
              VOICE THREAT INTELLIGENCE
            </div>
          </div>
        </div>

        <div style={{ width: 1, height: 28, background: 'var(--border)' }} />

        <span style={{
          fontSize: 10,
          fontWeight: 700,
          letterSpacing: '0.1em',
          padding: '3px 8px',
          background: 'rgba(16,185,129,0.1)',
          color: 'var(--green)',
          border: '1px solid rgba(16,185,129,0.3)',
          borderRadius: 4,
        }}>
          ● LOCAL INFERENCE
        </span>
      </div>

      {/* Right side */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 20 }}>
        {/* Model status */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <StatusDot ok={modelReady} pulse={!modelReady} />
          <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
            {modelReady ? 'Model Ready' : 'Model Offline'}
          </span>
        </div>

        {/* API status */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <StatusDot ok={health !== null} pulse={health === null} />
          <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
            {health !== null ? 'API Connected' : 'API Offline'}
          </span>
        </div>

        {/* Clock */}
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--text-muted)', minWidth: 80 }}>
          {now.toLocaleTimeString('en-US', { hour12: false })}
        </div>
      </div>
    </header>
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

  const ACCEPTED = ['audio/flac', 'audio/wav', 'audio/ogg', 'audio/mpeg', 'audio/mp4', 'audio/x-wav']
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
    <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      <SectionTitle icon="🎙">Live Audio Analysis</SectionTitle>

      {/* Drop zone */}
      <div
        onClick={() => inputRef.current?.click()}
        onDragOver={e => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        style={{
          border: `2px dashed ${dragging ? 'var(--accent)' : file ? 'var(--green)' : 'var(--border)'}`,
          borderRadius: 'var(--radius)',
          padding: '24px 16px',
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
            <div style={{ fontSize: 22, marginBottom: 4 }}>🎵</div>
            <div style={{ color: 'var(--text-primary)', fontWeight: 600, fontSize: 13 }}>
              {file.name}
            </div>
            <div style={{ color: 'var(--text-muted)', fontSize: 11, marginTop: 2 }}>
              {sizeKB} KB · {file.type || 'audio'}
            </div>
          </div>
        ) : (
          <div>
            <div style={{ fontSize: 22, marginBottom: 8 }}>⬆</div>
            <div style={{ color: 'var(--text-secondary)', fontSize: 13 }}>
              Drop audio file here or click to upload
            </div>
            <div style={{ color: 'var(--text-muted)', fontSize: 11, marginTop: 4 }}>
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
          padding: '8px 12px',
          fontSize: 12,
        }}>
          ⚠ {error}
        </div>
      )}

      {/* Analyze button */}
      <button
        className="btn btn-primary"
        style={{ width: '100%', padding: '10px', fontSize: 13 }}
        disabled={!file || loading || !modelReady}
        onClick={analyze}
        title={!modelReady ? 'Model offline — train first' : undefined}
      >
        {loading
          ? <><span className="spinner" style={{ width: 16, height: 16 }} /> Analyzing...</>
          : !modelReady
          ? '⚠ Model Offline — Training Required'
          : '⚡ Analyze Audio'
        }
      </button>

      {!modelReady && (
        <div style={{
          textAlign: 'center',
          color: 'var(--yellow)',
          fontSize: 11,
          padding: '4px 8px',
          background: 'var(--yellow-dim)',
          borderRadius: 'var(--radius-sm)',
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
          flex: 1, display: 'flex', flexDirection: 'column',
          alignItems: 'center', justifyContent: 'center',
          color: 'var(--text-muted)', fontSize: 13,
          padding: '24px 0',
          gap: 8,
        }}>
          <span style={{ fontSize: 32 }}>🔍</span>
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
          padding: 16,
          textAlign: 'center',
        }}>
          <div style={{ fontSize: 28, marginBottom: 8 }}>🔌</div>
          <div style={{ color: 'var(--yellow)', fontWeight: 700, fontSize: 14 }}>MODEL OFFLINE</div>
          <div style={{ color: 'var(--text-secondary)', fontSize: 12, marginTop: 4 }}>
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
          <div style={{ color: 'var(--red)', fontWeight: 600, fontSize: 13 }}>
            ⚠ {result.error}
          </div>
        </div>
      </div>
    )
  }

  const isSpoof = classification === 'SPOOF'
  const mainColor = isSpoof ? 'var(--red)' : 'var(--green)'
  const mainBg    = isSpoof ? 'var(--red-dim)' : 'var(--green-dim)'
  const spoofP = result.spoof_probability ?? 0
  const bfP    = result.bona_fide_probability ?? 0

  return (
    <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      <SectionTitle icon="📊">Analysis Result</SectionTitle>

      {/* Main verdict */}
      <div style={{
        background: mainBg,
        border: `1px solid ${mainColor}40`,
        borderRadius: 'var(--radius)',
        padding: '16px',
        textAlign: 'center',
        boxShadow: isSpoof ? '0 0 20px var(--red-glow)' : 'none',
      }}>
        <div style={{ fontSize: 28, marginBottom: 4 }}>
          {isSpoof ? '🚨' : '✅'}
        </div>
        <div style={{
          fontSize: 22,
          fontWeight: 800,
          color: mainColor,
          letterSpacing: '0.05em',
        }}>
          {isSpoof ? 'SYNTHETIC SPEECH' : 'BONA FIDE'}
        </div>
        <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 2 }}>
          {isSpoof ? 'Spoof / AI-generated audio detected' : 'Genuine human speech'}
        </div>
      </div>

      {/* Probabilities */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        {/* Spoof probability */}
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 3 }}>
            <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>Spoof Probability</span>
            <span style={{ fontSize: 12, fontFamily: 'var(--font-mono)', color: 'var(--red)' }}>
              {probabilityPct(spoofP)}
            </span>
          </div>
          <ProbBar value={spoofP} color="var(--red)" />
        </div>

        {/* Bona-fide probability */}
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 3 }}>
            <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>Bona-Fide Probability</span>
            <span style={{ fontSize: 12, fontFamily: 'var(--font-mono)', color: 'var(--green)' }}>
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
            padding: '8px 10px',
          }}>
            <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 2 }}>{label}</div>
            <div style={{ fontSize: 13, fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }}>{value}</div>
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
        <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>Threat Level</span>
        <ThreatBadge level={result.threat_level} />
      </div>

      {/* Recommended action */}
      <div style={{
        background: 'var(--bg-secondary)',
        border: '1px solid var(--border)',
        borderRadius: 'var(--radius-sm)',
        padding: '8px 12px',
      }}>
        <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 4 }}>
          RECOMMENDED ACTION
        </div>
        <div style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.5 }}>
          {result.recommended_action}
        </div>
      </div>

      {/* Footer */}
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: 'var(--text-muted)' }}>
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

  const ready   = modelInfo?.status === 'ready'
  const vm      = modelInfo?.val_metrics || {}
  const dsType  = modelInfo?.dataset_type || '—'
  const isReal  = dsType === 'ASVspoof5'

  return (
    <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <SectionTitle icon="🧠">Model</SectionTitle>
        <button
          className="btn btn-secondary"
          style={{ padding: '4px 8px', fontSize: 10 }}
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
          padding: '12px',
          textAlign: 'center',
        }}>
          <div style={{ color: 'var(--yellow)', fontWeight: 700, marginBottom: 4, fontSize: 13 }}>
            ⚠ MODEL OFFLINE
          </div>
          <div style={{ color: 'var(--text-muted)', fontSize: 11 }}>
            Training required
          </div>
          <div style={{ color: 'var(--text-muted)', fontSize: 10, marginTop: 4, fontFamily: 'var(--font-mono)' }}>
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

          {/* Model info */}
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
              <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 4, letterSpacing: '0.08em' }}>
                VALIDATION METRICS (from checkpoint)
              </div>
              <div style={{
                background: 'var(--bg-secondary)',
                borderRadius: 'var(--radius-sm)',
                padding: '8px 10px',
                fontSize: 11,
                display: 'grid',
                gridTemplateColumns: '1fr 1fr',
                gap: '4px 16px',
              }}>
                {[
                  ['ROC-AUC', vm.roc_auc],
                  ['EER',     vm.eer],
                  ['F1',      vm.f1],
                  ['Recall',  vm.recall],
                  ['FPR',     vm.fpr],
                  ['FNR',     vm.fnr],
                ].filter(([,v]) => v !== undefined).map(([k, v]) => (
                  <div key={k} style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: 'var(--text-muted)' }}>{k}</span>
                    <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>
                      {fmt(v, 4)}
                    </span>
                  </div>
                ))}
              </div>
              <div style={{ fontSize: 10, color: 'var(--text-muted)', lineHeight: 1.4 }}>
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
        <div style={{ color: 'var(--text-muted)', fontSize: 12, padding: '12px 0', textAlign: 'center' }}>
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
          const color = THREAT_COLORS[ev.threat_level] || 'var(--text-muted)'
          return (
            <div key={ev.id} style={{
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              padding: '5px 8px',
              borderRadius: 'var(--radius-sm)',
              background: 'var(--bg-secondary)',
              fontSize: 11,
            }}>
              <span style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', fontSize: 10, minWidth: 64 }}>
                {shortTs(ev.timestamp)}
              </span>
              <span style={{
                color: isSpoof ? 'var(--red)' : 'var(--green)',
                fontWeight: 700, fontSize: 10, minWidth: 64,
              }}>
                {ev.classification}
              </span>
              <div style={{ flex: 1 }}>
                <div style={{
                  height: 3, background: 'var(--border)', borderRadius: 2, overflow: 'hidden'
                }}>
                  <div style={{
                    height: '100%',
                    width: `${(ev.spoof_prob ?? 0) * 100}%`,
                    background: color,
                    borderRadius: 2,
                  }} />
                </div>
              </div>
              <span style={{ color, fontFamily: 'var(--font-mono)', fontSize: 10, minWidth: 36, textAlign: 'right' }}>
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
        <div style={{ color: 'var(--text-muted)', fontSize: 12, padding: '12px 0', textAlign: 'center' }}>
          No incidents logged yet
        </div>
      </div>
    )
  }

  return (
    <div className="card">
      <SectionTitle icon="🗂">Incident History</SectionTitle>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
          <thead>
            <tr style={{ borderBottom: '1px solid var(--border)' }}>
              {['Time', 'Classification', 'Spoof P', 'Risk', 'Threat', 'Latency', 'File'].map(h => (
                <th key={h} style={{
                  textAlign: 'left', padding: '6px 8px',
                  color: 'var(--text-muted)', fontWeight: 600, fontSize: 10,
                  letterSpacing: '0.05em', textTransform: 'uppercase',
                  whiteSpace: 'nowrap',
                }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {incidents.slice(0, 20).map((inc) => (
              <tr key={inc.id} style={{
                borderBottom: '1px solid var(--border)',
              }}>
                <td style={{ padding: '5px 8px', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', whiteSpace: 'nowrap' }}>
                  {shortTs(inc.timestamp)}
                </td>
                <td style={{ padding: '5px 8px' }}>
                  <span style={{
                    color: inc.classification === 'SPOOF' ? 'var(--red)' : 'var(--green)',
                    fontWeight: 700, fontSize: 10,
                  }}>
                    {inc.classification}
                  </span>
                </td>
                <td style={{ padding: '5px 8px', fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>
                  {inc.spoof_prob !== null && inc.spoof_prob !== undefined
                    ? probabilityPct(inc.spoof_prob)
                    : '—'
                  }
                </td>
                <td style={{ padding: '5px 8px', fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>
                  {inc.risk_score !== null && inc.risk_score !== undefined
                    ? inc.risk_score.toFixed(3)
                    : '—'
                  }
                </td>
                <td style={{ padding: '5px 8px' }}>
                  <ThreatBadge level={inc.threat_level} />
                </td>
                <td style={{ padding: '5px 8px', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
                  {inc.latency_ms !== null ? `${inc.latency_ms.toFixed(0)}ms` : '—'}
                </td>
                <td style={{ padding: '5px 8px', color: 'var(--text-muted)', maxWidth: 120 }} className="truncate">
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
// Stats Panel
// ──────────────────────────────────────────────────────────────────────────────

function StatsPanel({ stats }) {
  if (!stats || stats.total === 0) {
    return (
      <div className="card">
        <SectionTitle icon="📈">Statistics</SectionTitle>
        <div style={{ color: 'var(--text-muted)', fontSize: 12, textAlign: 'center', padding: '12px 0' }}>
          No data yet
        </div>
      </div>
    )
  }

  const byThr = stats.by_threat || {}

  return (
    <div className="card">
      <SectionTitle icon="📈">Statistics</SectionTitle>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8, marginBottom: 12 }}>
        {[
          { label: 'Total', value: stats.total },
          { label: 'Spoof Rate', value: stats.spoof_rate !== null ? pct(stats.spoof_rate) : '—' },
          { label: 'Avg Latency', value: stats.avg_latency_ms ? `${stats.avg_latency_ms}ms` : '—' },
        ].map(({ label, value }) => (
          <div key={label} style={{
            background: 'var(--bg-secondary)',
            border: '1px solid var(--border)',
            borderRadius: 'var(--radius-sm)',
            padding: '8px',
            textAlign: 'center',
          }}>
            <div style={{ fontSize: 18, fontWeight: 700, fontFamily: 'var(--font-mono)' }}>{value}</div>
            <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>{label}</div>
          </div>
        ))}
      </div>

      {/* Threat breakdown */}
      {Object.keys(byThr).length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          {['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'].filter(l => byThr[l]).map(level => (
            <div key={level} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ minWidth: 64, fontSize: 10 }}>
                <ThreatBadge level={level} />
              </span>
              <div className="progress-bar" style={{ flex: 1 }}>
                <div className="progress-bar-fill" style={{
                  width: `${(byThr[level] / stats.total) * 100}%`,
                  background: THREAT_COLORS[level],
                }} />
              </div>
              <span style={{ fontSize: 11, fontFamily: 'var(--font-mono)', minWidth: 24, textAlign: 'right', color: 'var(--text-secondary)' }}>
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
  const ready = modelInfo?.status === 'ready'
  const checks = [
    { label: 'API Server',      ok: health !== null,      value: health ? 'Connected' : 'Offline' },
    { label: 'Model',           ok: ready,                value: ready ? 'Loaded' : 'Offline' },
    { label: 'Inference',       ok: ready,                value: health?.inference_mode || '—' },
    { label: 'Device',          ok: true,                 value: health?.device || '—' },
    { label: 'External APIs',   ok: true,                 value: 'NONE' },
    { label: 'Pretrained Model',ok: true,                 value: 'No' },
  ]

  return (
    <div className="card">
      <SectionTitle icon="🖥">System Status</SectionTitle>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
        {checks.map(({ label, ok, value }) => (
          <div key={label} style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: '4px 0',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <StatusDot ok={ok} />
              <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>{label}</span>
            </div>
            <span style={{
              fontSize: 11,
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
        fontSize: 10,
        color: 'var(--text-muted)',
        lineHeight: 1.5,
        background: 'var(--bg-secondary)',
        borderRadius: 'var(--radius-sm)',
        padding: '8px',
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
          padding: '8px 24px',
          color: 'var(--red)',
          fontSize: 12,
          textAlign: 'center',
        }}>
          ⚠ Cannot connect to VoxShield API (localhost:8000) — Start the backend with:
          <code style={{ marginLeft: 8, fontFamily: 'var(--font-mono)' }}>
            uvicorn backend.app:app --port 8000
          </code>
        </div>
      )}

      {/* Main content */}
      <div style={{
        flex: 1,
        display: 'grid',
        gridTemplateColumns: '320px 1fr 280px',
        gridTemplateRows: 'auto 1fr',
        gap: 16,
        padding: 16,
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
          <ModelPanel
            modelInfo={modelInfo}
            onReload={refreshData}
          />
          <StatsPanel stats={stats} />
          <SystemStatus health={health} modelInfo={modelInfo} />
        </div>
      </div>

      {/* Footer */}
      <footer style={{
        borderTop: '1px solid var(--border)',
        padding: '8px 24px',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        fontSize: 10,
        color: 'var(--text-muted)',
      }}>
        <span>VoxShield Independent v1.0 · Voice Threat Intelligence · Local Inference Only</span>
        <span>VoxShieldNet trained from random initialization · No pretrained components · No external APIs</span>
        <span>© 2026 · SIH Demo Build</span>
      </footer>
    </div>
  )
}
