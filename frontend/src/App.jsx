/**
 * VoxShield SOC Dashboard — Sidebar Layout
 * Voice Threat Intelligence — Local Inference Only
 *
 * Layout: left sidebar (256 px) + topbar + main content area
 * Reference: VoxShield_Frontend_Reference_UI(1)/src/main.jsx
 * Typography: Inter, CSS variables from index.css
 */

import { useState, useEffect, useRef, useCallback } from 'react'
import {
  Activity, BarChart3, Bell, CheckCircle2, ChevronDown,
  Database, LayoutDashboard, Mic, Settings, Shield,
  ShieldAlert, Users, Waves,
} from 'lucide-react'
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
// Reusable primitives
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
      fontSize: 'var(--fs-xxs)',
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
 * label: 12px muted  |  value: 13px primary
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
 * SectionTitle — uppercase card section label.
 * 11px, 600 weight — intentionally small as a section divider.
 */
function SectionTitle({ children, icon }) {
  return (
    <div style={{
      fontSize: 'var(--fs-xxs)',
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
// Sidebar
// ──────────────────────────────────────────────────────────────────────────────

const NAV_ITEMS = [
  ['Dashboard',         LayoutDashboard],
  ['Analyze Voice',     Activity],
  ['Live Monitor',      Mic],
  ['Incidents',         Database],
  ['Analytics',         BarChart3],
  ['Model Performance', Waves],
  ['Settings',          Settings],
]

function Logo() {
  return (
    <div style={{
      height: 64,
      display: 'flex',
      gap: 12,
      alignItems: 'center',
      padding: '0 8px 10px',
      borderBottom: '1px solid rgba(255,255,255,0.06)',
      marginBottom: 6,
    }}>
      {/* Shield icon */}
      <div style={{
        width: 43, height: 47,
        color: '#596dff',
        display: 'grid',
        placeItems: 'center',
        border: '3px solid #4f61e8',
        clipPath: 'polygon(50% 0,96% 17%,90% 70%,50% 100%,10% 70%,4% 17%)',
        background: 'rgba(56,77,220,0.08)',
        flexShrink: 0,
      }}>
        <Waves size={20} />
      </div>
      <div>
        <div style={{
          fontSize: 22,
          fontWeight: 800,
          letterSpacing: '-0.5px',
          color: 'var(--text-primary)',
          lineHeight: 1.2,
        }}>
          VoxShield
        </div>
        <div style={{
          fontSize: 9,
          color: 'var(--text-muted)',
          letterSpacing: '1.7px',
          textTransform: 'uppercase',
          marginTop: 1,
        }}>
          TRUST EVERY VOICE
        </div>
      </div>
    </div>
  )
}

function Sidebar({ active, setActive }) {
  return (
    <aside style={{
      width: 240,
      minWidth: 240,
      borderRight: '1px solid var(--border)',
      background: 'linear-gradient(180deg, #09121f, #07101a)',
      padding: '17px 14px 20px',
      display: 'flex',
      flexDirection: 'column',
      position: 'fixed',
      inset: '0 auto 0 0',
      overflowY: 'auto',
      zIndex: 100,
    }}>
      <Logo />

      <nav style={{ marginTop: 12 }}>
        {NAV_ITEMS.map(([label, Icon]) => {
          const isActive = active === label
          return (
            <button
              key={label}
              onClick={() => setActive(label)}
              style={{
                width: '100%',
                border: isActive ? 'none' : 'none',
                background: isActive
                  ? 'linear-gradient(90deg, rgba(54,75,125,0.5), rgba(39,57,92,0.3))'
                  : 'transparent',
                color: isActive ? '#fff' : 'var(--text-secondary)',
                height: 48,
                borderRadius: 9,
                display: 'flex',
                alignItems: 'center',
                gap: 14,
                padding: '0 14px',
                margin: '2px 0',
                fontSize: 'var(--fs-sm)',
                textAlign: 'left',
                cursor: 'pointer',
                transition: 'all 0.15s ease',
                boxShadow: isActive ? 'inset 3px 0 0 #5068ff' : 'none',
              }}
            >
              <Icon size={19} />
              <span>{label}</span>
            </button>
          )
        })}
      </nav>

      {/* India card */}
      <div style={{
        marginTop: 'auto',
        border: '1px solid var(--border)',
        borderRadius: 8,
        background: 'linear-gradient(145deg, #111d2e, #0c1725)',
        padding: 14,
        display: 'flex',
        gap: 10,
        alignItems: 'flex-start',
      }}>
        <div style={{ fontSize: 22, flexShrink: 0 }}>🇮🇳</div>
        <div>
          <div style={{ fontSize: 12, fontWeight: 700, lineHeight: 1.45, color: 'var(--text-primary)' }}>
            Built for a<br />Safer India
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.55, marginTop: 5 }}>
            Combating voice fraud<br />with AI
          </div>
        </div>
      </div>
    </aside>
  )
}

// ──────────────────────────────────────────────────────────────────────────────
// Topbar
// ──────────────────────────────────────────────────────────────────────────────

function Topbar({ modelReady }) {
  const [now, setNow] = useState(new Date())
  useEffect(() => {
    const t = setInterval(() => setNow(new Date()), 1000)
    return () => clearInterval(t)
  }, [])

  return (
    <header style={{
      height: 64,
      borderBottom: '1px solid var(--border)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      padding: '0 28px',
      background: 'rgba(5,11,19,0.85)',
      backdropFilter: 'blur(8px)',
      flexShrink: 0,
    }}>
      {/* Left */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, color: 'var(--text-secondary)', fontSize: 'var(--fs-sm)' }}>
        <Shield size={17} style={{ color: 'var(--green)' }} />
        <span>AI-Powered Voice Security for a Safer Tomorrow</span>
      </div>

      {/* Right */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 20 }}>
        {/* Online status */}
        <span style={{
          display: 'flex', gap: 6, alignItems: 'center',
          color: 'var(--green)', fontSize: 'var(--fs-xs)',
        }}>
          <span className="pulse" style={{
            width: 8, height: 8, borderRadius: '50%',
            background: 'var(--green)',
            display: 'inline-block',
          }} />
          System Online
        </span>

        {/* Model status */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <StatusDot ok={modelReady} pulse={!modelReady} />
          <span style={{ fontSize: 'var(--fs-xs)', color: 'var(--text-secondary)' }}>
            {modelReady ? 'Model Ready' : 'Model Offline'}
          </span>
        </div>

        {/* Clock */}
        <div style={{
          fontFamily: 'var(--font-mono)',
          fontSize: 'var(--fs-xs)',
          color: 'var(--text-muted)',
          minWidth: 68,
          letterSpacing: '0.04em',
        }}>
          {now.toLocaleTimeString('en-US', { hour12: false })}
        </div>

        {/* Bell */}
        <Bell size={18} style={{ color: 'var(--text-muted)', cursor: 'pointer' }} />

        {/* Profile */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 9, cursor: 'pointer' }}>
          <div style={{
            width: 36, height: 36, borderRadius: '50%',
            display: 'grid', placeItems: 'center',
            fontSize: 16, color: '#ff9da3',
            background: 'linear-gradient(145deg, #252b39, #151b27)',
          }}>
            V
          </div>
          <div>
            <div style={{ fontSize: 'var(--fs-xs)', fontWeight: 600, color: 'var(--text-primary)', lineHeight: 1.2 }}>Admin</div>
            <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>Team Admin</div>
          </div>
          <ChevronDown size={14} style={{ color: 'var(--text-muted)' }} />
        </div>
      </div>
    </header>
  )
}

// ──────────────────────────────────────────────────────────────────────────────
// Dashboard page — hero + stat cards + main grid
// ──────────────────────────────────────────────────────────────────────────────

function DashboardHero({ modelReady }) {
  return (
    <div style={{ padding: '24px 24px 0', marginBottom: 4 }}>
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
      <p style={{
        fontSize: 'var(--fs-body)',
        color: 'var(--text-secondary)',
        marginTop: 6,
        lineHeight: 'var(--lh-body)',
      }}>
        Detect. Analyze. Prevent. Build Trust in Every Conversation.
        {!modelReady && (
          <span style={{ marginLeft: 10, color: 'var(--yellow)', fontWeight: 500 }}>
            ⚠ Model offline — training required.
          </span>
        )}
      </p>
    </div>
  )
}

function StatCards({ stats }) {
  const total   = stats?.total ?? 0
  const spoof   = stats?.by_classification?.SPOOF ?? 0
  const blocked   = spoof
  const trusted   = stats?.by_classification?.BONA_FIDE ?? (total - spoof)

  const cards = [
    { icon: Waves,        label: 'Total Analyses',  value: total,   kind: 'blue'  },
    { icon: ShieldAlert,  label: 'Threats Detected', value: spoof,  kind: 'red'   },
    { icon: CheckCircle2, label: 'Blocked Attempts', value: blocked, kind: 'green' },
    { icon: Users,        label: 'Trusted Calls',    value: trusted, kind: 'blue'  },
  ]

  const iconColors = {
    blue:  { color: '#5c78ff', bg: 'rgba(63,91,220,0.18)'  },
    red:   { color: '#ff5960', bg: 'rgba(219,58,67,0.17)'  },
    green: { color: '#2ce2a4', bg: 'rgba(30,194,133,0.17)' },
  }

  return (
    <div style={{
      display: 'grid',
      gridTemplateColumns: 'repeat(4, 1fr)',
      gap: 14,
      padding: '16px 24px',
    }}>
      {cards.map(({ icon: Icon, label, value, kind }) => {
        const { color, bg } = iconColors[kind]
        return (
          <div key={label} style={{
            height: 84,
            border: '1px solid var(--border)',
            borderRadius: 10,
            background: 'linear-gradient(145deg, var(--bg-card), var(--bg-secondary))',
            display: 'flex',
            alignItems: 'center',
            padding: '10px 14px',
            gap: 12,
            boxShadow: 'var(--shadow)',
          }}>
            <div style={{
              width: 48, height: 48, borderRadius: 12,
              display: 'grid', placeItems: 'center',
              color, background: bg, flexShrink: 0,
            }}>
              <Icon size={22} />
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', flex: 1, minWidth: 0 }}>
              <span style={{ fontSize: 'var(--fs-xs)', color: 'var(--text-muted)' }}>{label}</span>
              {/* KPI value — 28px 700 */}
              <strong style={{
                fontSize: 28,
                fontWeight: 700,
                lineHeight: 1.25,
                marginTop: 2,
                fontFamily: 'var(--font-mono)',
                color: 'var(--text-primary)',
              }}>{value}</strong>
            </div>
          </div>
        )
      })}
    </div>
  )
}

// ──────────────────────────────────────────────────────────────────────────────
// Upload + Analysis Panel
// ──────────────────────────────────────────────────────────────────────────────

function UploadPanel({ onResult, modelReady }) {
  const [file, setFile]         = useState(null)
  const [dragging, setDragging] = useState(false)
  const [loading, setLoading]   = useState(false)
  const [error, setError]       = useState(null)
  const inputRef = useRef(null)
  // dragCounter tracks nested dragenter/dragleave so we don't flicker when
  // the cursor moves over child elements inside the drop zone.
  const dragCounter = useRef(0)

  const ACCEPTED_EXT = ['.flac', '.wav', '.ogg', '.mp3', '.m4a']

  function handleFile(f) {
    if (!f) return
    setError(null)
    setFile(f)
  }

  function clearFile(e) {
    e.stopPropagation()
    setFile(null)
    setError(null)
    // Reset the hidden file input so the same file can be re-selected.
    if (inputRef.current) inputRef.current.value = ''
  }

  // ── Drag-and-drop handlers ─────────────────────────────────────────────────
  function onDragEnter(e) {
    e.preventDefault()
    dragCounter.current += 1
    if (dragCounter.current === 1) setDragging(true)
  }

  function onDragOver(e) {
    // Must call preventDefault() to allow drop.
    e.preventDefault()
  }

  function onDragLeave(e) {
    e.preventDefault()
    dragCounter.current -= 1
    if (dragCounter.current === 0) setDragging(false)
  }

  function onDrop(e) {
    e.preventDefault()
    dragCounter.current = 0
    setDragging(false)
    const f = e.dataTransfer.files[0]
    if (f) handleFile(f)
  }

  // ── Analyze ────────────────────────────────────────────────────────────────
  async function analyze() {
    if (!file || loading) return
    setLoading(true)
    setError(null)
    try {
      const result = await api.predict(file)
      // Attach the local filename so the ResultPanel can display it.
      onResult({ ...result, _filename: file.name })
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const sizeKB  = file ? (file.size / 1024).toFixed(1) : null
  const sizeMB  = file ? (file.size / (1024 * 1024)).toFixed(2) : null
  const fileExt = file ? ('.' + file.name.split('.').pop().toLowerCase()) : null
  const extOk   = fileExt ? ACCEPTED_EXT.includes(fileExt) : true

  return (
    <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
      <SectionTitle icon="🎙">Analyze Voice</SectionTitle>
      <p style={{ fontSize: 'var(--fs-xs)', color: 'var(--text-muted)', marginTop: -8, marginBottom: 4, lineHeight: 'var(--lh-compact)' }}>
        Upload an audio file to detect deepfake and cloned voices.
      </p>

      {/* Drop zone */}
      <div
        role="button"
        tabIndex={0}
        aria-label="Click or drop audio file here"
        onClick={() => !loading && inputRef.current?.click()}
        onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); !loading && inputRef.current?.click() } }}
        onDragEnter={onDragEnter}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
        style={{
          border: `2px dashed ${dragging ? 'var(--accent)' : file ? (extOk ? 'var(--green)' : 'var(--red)') : 'var(--border)'}`,
          borderRadius: 'var(--radius)',
          padding: '24px 16px',
          textAlign: 'center',
          cursor: loading ? 'not-allowed' : 'pointer',
          background: dragging ? 'var(--accent-glow)' : file ? (extOk ? 'var(--green-dim)' : 'var(--red-dim)') : 'transparent',
          transition: 'var(--transition)',
          userSelect: 'none',
          outline: 'none',
          position: 'relative',
        }}
      >
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED_EXT.join(',')}
          style={{ display: 'none' }}
          onChange={e => { if (e.target.files[0]) handleFile(e.target.files[0]) }}
        />
        {file ? (
          <div style={{ position: 'relative' }}>
            {/* Clear / replace button */}
            <button
              aria-label="Remove selected file"
              onClick={clearFile}
              style={{
                position: 'absolute',
                top: -8,
                right: -8,
                width: 22,
                height: 22,
                borderRadius: '50%',
                background: 'var(--bg-card)',
                border: '1px solid var(--border)',
                color: 'var(--text-muted)',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: 12,
                lineHeight: 1,
                padding: 0,
                zIndex: 1,
              }}
            >
              ✕
            </button>
            <div style={{ fontSize: 24, marginBottom: 6 }}>🎵</div>
            <div style={{
              color: 'var(--text-primary)',
              fontWeight: 600,
              fontSize: 'var(--fs-body)',
              lineHeight: 'var(--lh-compact)',
              wordBreak: 'break-all',
            }}>
              {file.name}
            </div>
            <div style={{
              color: 'var(--text-muted)',
              fontSize: 'var(--fs-xs)',
              marginTop: 3,
            }}>
              {parseFloat(sizeKB) >= 1024 ? `${sizeMB} MB` : `${sizeKB} KB`}
              {' · '}
              {file.type || fileExt || 'audio'}
            </div>
            {!extOk && (
              <div style={{ color: 'var(--red)', fontSize: 'var(--fs-xs)', marginTop: 4, fontWeight: 500 }}>
                Unsupported format. Use: {ACCEPTED_EXT.join(' ')}
              </div>
            )}
            <div style={{ color: 'var(--text-muted)', fontSize: 'var(--fs-xs)', marginTop: 6 }}>
              Click to replace
            </div>
          </div>
        ) : (
          <div>
            <div style={{ fontSize: 24, marginBottom: 8 }}>⬆</div>
            <div style={{
              color: dragging ? 'var(--accent-hover)' : 'var(--text-secondary)',
              fontSize: 'var(--fs-body)',
              fontWeight: 500,
            }}>
              {dragging ? 'Drop file here' : 'Drop audio file here or click to upload'}
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
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'flex-start',
          gap: 8,
        }}>
          <span>⚠ {error}</span>
          <button
            onClick={() => setError(null)}
            style={{
              background: 'none',
              border: 'none',
              color: 'var(--red)',
              cursor: 'pointer',
              fontSize: 14,
              padding: 0,
              flexShrink: 0,
              lineHeight: 1,
            }}
            aria-label="Dismiss error"
          >
            ✕
          </button>
        </div>
      )}

      {/* Analyze button */}
      <button
        className="btn btn-primary"
        style={{ width: '100%', padding: '11px' }}
        disabled={!file || loading || !modelReady || !extOk}
        onClick={analyze}
        title={
          !modelReady  ? 'Model offline — start the backend server' :
          !file        ? 'Select an audio file first' :
          !extOk       ? `Unsupported format — use ${ACCEPTED_EXT.join(', ')}` :
          undefined
        }
      >
        {loading
          ? <><span className="spinner" /> Analyzing {file?.name}…</>
          : !modelReady
          ? '⚠ Model Offline — Start Backend'
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
          MODEL OFFLINE — start backend: <code style={{ fontFamily: 'var(--font-mono)' }}>uvicorn backend.app:app --port 8000</code>
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
      <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: 10, minHeight: 180 }}>
        <SectionTitle icon="📊">Detection Result</SectionTitle>
        <div style={{
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          color: 'var(--text-muted)',
          fontSize: 'var(--fs-sm)',
          padding: '20px 0',
          gap: 10,
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
        <SectionTitle icon="⚠">Detection Result</SectionTitle>
        <div style={{
          background: 'var(--yellow-dim)',
          border: '1px solid rgba(245,158,11,0.3)',
          borderRadius: 'var(--radius)',
          padding: 20,
          textAlign: 'center',
        }}>
          <div style={{ fontSize: 28, marginBottom: 10 }}>🔌</div>
          <div style={{ color: 'var(--yellow)', fontWeight: 700, fontSize: 'var(--fs-body)' }}>
            MODEL OFFLINE
          </div>
          <div style={{ color: 'var(--text-secondary)', fontSize: 'var(--fs-sm)', marginTop: 6, lineHeight: 'var(--lh-body)' }}>
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
          <div style={{ color: 'var(--red)', fontWeight: 600, fontSize: 'var(--fs-sm)', lineHeight: 'var(--lh-compact)' }}>
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
      <SectionTitle icon="📊">Detection Result ✓ Completed</SectionTitle>

      {/* Main verdict */}
      <div style={{
        background: mainBg,
        border: `1px solid ${mainColor}40`,
        borderRadius: 'var(--radius)',
        padding: '18px 16px',
        textAlign: 'center',
        boxShadow: isSpoof ? '0 0 20px var(--red-glow)' : 'none',
      }}>
        <div style={{ fontSize: 28, marginBottom: 6 }}>{isSpoof ? '🚨' : '✅'}</div>
        {/* Verdict — 24px 800 dominant */}
        <div style={{
          fontSize: 24,
          fontWeight: 800,
          color: mainColor,
          letterSpacing: '0.04em',
          lineHeight: 1.2,
        }}>
          {isSpoof ? 'SYNTHETIC SPEECH' : 'BONA FIDE'}
        </div>
        <div style={{ fontSize: 'var(--fs-sm)', color: 'var(--text-secondary)', marginTop: 5 }}>
          {isSpoof ? 'Spoof / AI-generated audio detected' : 'Genuine human speech'}
        </div>
      </div>

      {/* Probabilities */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
            <span style={{ fontSize: 'var(--fs-xs)', color: 'var(--text-muted)' }}>Spoof Probability</span>
            <span style={{ fontSize: 'var(--fs-sm)', fontFamily: 'var(--font-mono)', color: 'var(--red)', fontWeight: 600 }}>
              {probabilityPct(spoofP)}
            </span>
          </div>
          <ProbBar value={spoofP} color="var(--red)" />
        </div>
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
            <span style={{ fontSize: 'var(--fs-xs)', color: 'var(--text-muted)' }}>Bona-Fide Probability</span>
            <span style={{ fontSize: 'var(--fs-sm)', fontFamily: 'var(--font-mono)', color: 'var(--green)', fontWeight: 600 }}>
              {probabilityPct(bfP)}
            </span>
          </div>
          <ProbBar value={bfP} color="var(--green)" />
        </div>
      </div>

      {/* Metrics grid */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
        {[
          { label: 'Confidence', value: probabilityPct(result.confidence) },
          { label: 'Risk Score', value: fmt(result.risk_score, 3) },
          { label: 'Threshold',  value: fmt(result.decision_threshold, 4) },
          { label: 'Latency',    value: `${fmt(result.latency_ms, 1)} ms` },
        ].map(({ label, value }) => (
          <div key={label} style={{
            background: 'var(--bg-secondary)',
            border: '1px solid var(--border)',
            borderRadius: 'var(--radius-sm)',
            padding: '9px 11px',
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

      {/* Threat level + recommended action */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '6px 0' }}>
        <span style={{ fontSize: 'var(--fs-sm)', color: 'var(--text-secondary)', fontWeight: 500 }}>Threat Level</span>
        <ThreatBadge level={result.threat_level} />
      </div>
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
        <div style={{ fontSize: 'var(--fs-sm)', color: 'var(--text-secondary)', lineHeight: 'var(--lh-body)' }}>
          {result.recommended_action}
        </div>
      </div>

      {/* Footer meta */}
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 'var(--fs-xxs)', color: 'var(--text-muted)', flexWrap: 'wrap', gap: 4 }}>
        {result._filename && (
          <span style={{ flex: '0 0 100%', marginBottom: 2, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            File: {result._filename}
          </span>
        )}
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
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <SectionTitle icon="🧠">Model Performance</SectionTitle>
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
          padding: 14,
          textAlign: 'center',
        }}>
          <div style={{ color: 'var(--yellow)', fontWeight: 700, marginBottom: 5, fontSize: 'var(--fs-body)' }}>
            ⚠ MODEL OFFLINE
          </div>
          <div style={{ color: 'var(--text-muted)', fontSize: 'var(--fs-sm)' }}>Training required</div>
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
                ].filter(([, v]) => v !== undefined).map(([k, v]) => (
                  <div key={k} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
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
// Timeline Panel (Recent Detection Events)
// ──────────────────────────────────────────────────────────────────────────────

function TimelinePanel({ timeline }) {
  if (!timeline || timeline.length === 0) {
    return (
      <div className="card">
        <SectionTitle icon="📡">Recent Detection Events</SectionTitle>
        <div style={{ color: 'var(--text-muted)', fontSize: 'var(--fs-sm)', padding: '14px 0', textAlign: 'center' }}>
          No analysis events yet
        </div>
      </div>
    )
  }

  return (
    <div className="card">
      <SectionTitle icon="📡">Recent Detection Events</SectionTitle>
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
              <span style={{
                color: 'var(--text-muted)',
                fontFamily: 'var(--font-mono)',
                fontSize: 'var(--fs-xs)',
                minWidth: 68,
                flexShrink: 0,
              }}>
                {shortTs(ev.timestamp)}
              </span>
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
              <div style={{ flex: 1 }}>
                <div style={{ height: 3, background: 'var(--border)', borderRadius: 2, overflow: 'hidden' }}>
                  <div style={{
                    height: '100%',
                    width: `${(ev.spoof_prob ?? 0) * 100}%`,
                    background: color,
                    borderRadius: 2,
                  }} />
                </div>
              </div>
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
        <SectionTitle icon="🗂">Recent Incidents</SectionTitle>
        <div style={{ color: 'var(--text-muted)', fontSize: 'var(--fs-sm)', padding: '14px 0', textAlign: 'center' }}>
          No incidents logged yet
        </div>
      </div>
    )
  }

  return (
    <div className="card">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
        <SectionTitle icon="🗂">Recent Incidents</SectionTitle>
        <span style={{ fontSize: 'var(--fs-xxs)', color: 'var(--text-muted)' }}>
          {incidents.length} logged
        </span>
      </div>
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
                <td style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', whiteSpace: 'nowrap', fontSize: 'var(--fs-xs)' }}>
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
                <td style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)', fontSize: 'var(--fs-sm)' }}>
                  {inc.spoof_prob !== null && inc.spoof_prob !== undefined ? probabilityPct(inc.spoof_prob) : '—'}
                </td>
                <td style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)', fontSize: 'var(--fs-sm)' }}>
                  {inc.risk_score !== null && inc.risk_score !== undefined ? inc.risk_score.toFixed(3) : '—'}
                </td>
                <td><ThreatBadge level={inc.threat_level} /></td>
                <td style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', fontSize: 'var(--fs-xs)', whiteSpace: 'nowrap' }}>
                  {inc.latency_ms !== null ? `${inc.latency_ms.toFixed(0)}ms` : '—'}
                </td>
                <td style={{ color: 'var(--text-muted)', maxWidth: 120, fontSize: 'var(--fs-sm)' }} className="truncate">
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
        <div style={{ color: 'var(--text-muted)', fontSize: 'var(--fs-sm)', textAlign: 'center', padding: '14px 0' }}>
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
            {/* KPI number — 28px 700 */}
            <div style={{
              fontSize: 28,
              fontWeight: 700,
              fontFamily: 'var(--font-mono)',
              color: 'var(--text-primary)',
              lineHeight: 1.2,
            }}>
              {value}
            </div>
            <div style={{ fontSize: 'var(--fs-xs)', color: 'var(--text-muted)', marginTop: 4, fontWeight: 500 }}>
              {label}
            </div>
          </div>
        ))}
      </div>

      {/* Threat breakdown bars */}
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
              <span style={{ fontSize: 'var(--fs-sm)', color: 'var(--text-secondary)' }}>{label}</span>
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
// Page views (rendered based on active nav item)
// ──────────────────────────────────────────────────────────────────────────────

function DashboardPage({ health, modelInfo, result, timeline, incidents, stats, onResult, onReload }) {
  const modelReady = modelInfo?.status === 'ready'
  return (
    <>
      <DashboardHero modelReady={modelReady} />
      <StatCards stats={stats} />

      <div style={{
        display: 'grid',
        gridTemplateColumns: '340px 1fr 300px',
        gap: 16,
        padding: '0 24px 24px',
        alignItems: 'start',
      }}>
        {/* LEFT: upload + result */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <UploadPanel onResult={onResult} modelReady={modelReady} />
          <ResultPanel result={result} />
        </div>

        {/* MIDDLE: timeline + incidents */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16, minWidth: 0 }}>
          <TimelinePanel timeline={timeline} />
          <IncidentsPanel incidents={incidents} />
        </div>

        {/* RIGHT: model + stats + system */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <ModelPanel modelInfo={modelInfo} onReload={onReload} />
          <StatsPanel stats={stats} />
          <SystemStatus health={health} modelInfo={modelInfo} />
        </div>
      </div>
    </>
  )
}

function AnalyzeVoicePage({ modelInfo, result, onResult }) {
  const modelReady = modelInfo?.status === 'ready'
  return (
    <div style={{ padding: '24px' }}>
      <h1 style={{ fontSize: 'var(--fs-hero)', fontWeight: 700, marginBottom: 6, color: 'var(--text-primary)' }}>
        Analyze Voice
      </h1>
      <p style={{ fontSize: 'var(--fs-body)', color: 'var(--text-secondary)', marginBottom: 20 }}>
        Upload an audio file to detect deepfake and cloned voices.
      </p>
      <div style={{
        display: 'grid',
        gridTemplateColumns: result ? '1fr 1fr' : '1fr',
        gap: 20,
        maxWidth: result ? 1100 : 560,
        transition: 'max-width 0.3s ease',
      }}>
        <UploadPanel onResult={onResult} modelReady={modelReady} />
        {result && <ResultPanel result={result} />}
      </div>
    </div>
  )
}

function LiveMonitorPage() {
  return (
    <div style={{ padding: '24px' }}>
      <h1 style={{ fontSize: 'var(--fs-hero)', fontWeight: 700, marginBottom: 6, color: 'var(--text-primary)' }}>
        Live Monitor
      </h1>
      <p style={{ fontSize: 'var(--fs-body)', color: 'var(--text-secondary)', marginBottom: 20, lineHeight: 'var(--lh-body)' }}>
        The live microphone detection widget is running in the bottom-right corner.
        Click <strong style={{ color: 'var(--text-primary)' }}>🎙 Start Live Detection</strong> to begin real-time analysis.
      </p>
      <div style={{
        border: '1px solid var(--border)',
        borderRadius: 'var(--radius-lg)',
        padding: '20px 24px',
        background: 'var(--bg-card)',
        maxWidth: 560,
      }}>
        <div style={{ fontSize: 'var(--fs-sm)', color: 'var(--text-secondary)', lineHeight: 'var(--lh-body)' }}>
          <div style={{ marginBottom: 10, fontWeight: 600, color: 'var(--text-primary)' }}>Live Detection features:</div>
          {[
            'Rolling 4-second audio window at 16 kHz',
            'New verdict every ~2 seconds',
            'EMA-smoothed probability (α=0.35, window=5)',
            'Raw per-window spoof probability displayed unchanged',
            'All inference runs locally — no audio sent externally',
            'Session-tagged incidents logged to incident history',
          ].map(item => (
            <div key={item} style={{ display: 'flex', gap: 8, padding: '4px 0' }}>
              <span style={{ color: 'var(--green)', flexShrink: 0 }}>✓</span>
              <span>{item}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

function IncidentsPage({ incidents }) {
  return (
    <div style={{ padding: '24px' }}>
      <h1 style={{ fontSize: 'var(--fs-hero)', fontWeight: 700, marginBottom: 6, color: 'var(--text-primary)' }}>
        Incidents
      </h1>
      <p style={{ fontSize: 'var(--fs-body)', color: 'var(--text-secondary)', marginBottom: 20 }}>
        Full incident log from all analyses.
      </p>
      <IncidentsPanel incidents={incidents} />
    </div>
  )
}

function AnalyticsPage({ stats, timeline }) {
  return (
    <div style={{ padding: '24px' }}>
      <h1 style={{ fontSize: 'var(--fs-hero)', fontWeight: 700, marginBottom: 6, color: 'var(--text-primary)' }}>
        Analytics
      </h1>
      <p style={{ fontSize: 'var(--fs-body)', color: 'var(--text-secondary)', marginBottom: 20 }}>
        Aggregate statistics and threat distribution.
      </p>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, maxWidth: 900 }}>
        <StatsPanel stats={stats} />
        <TimelinePanel timeline={timeline} />
      </div>
    </div>
  )
}

function ModelPerformancePage({ modelInfo, onReload }) {
  return (
    <div style={{ padding: '24px' }}>
      <h1 style={{ fontSize: 'var(--fs-hero)', fontWeight: 700, marginBottom: 6, color: 'var(--text-primary)' }}>
        Model Performance
      </h1>
      <p style={{ fontSize: 'var(--fs-body)', color: 'var(--text-secondary)', marginBottom: 20 }}>
        VoxShieldNet validation metrics and model information.
      </p>
      <div style={{ maxWidth: 480 }}>
        <ModelPanel modelInfo={modelInfo} onReload={onReload} />
      </div>
    </div>
  )
}

function SettingsPage({ health }) {
  return (
    <div style={{ padding: '24px' }}>
      <h1 style={{ fontSize: 'var(--fs-hero)', fontWeight: 700, marginBottom: 6, color: 'var(--text-primary)' }}>
        Settings
      </h1>
      <p style={{ fontSize: 'var(--fs-body)', color: 'var(--text-secondary)', marginBottom: 20 }}>
        Backend configuration and system information.
      </p>
      <div className="card" style={{ maxWidth: 480 }}>
        <SectionTitle icon="⚙">Connection</SectionTitle>
        <MetricRow label="Backend URL"  value="http://localhost:8000" />
        <MetricRow label="API Status"   value={health ? 'Connected' : 'Offline'} mono={false} />
        <MetricRow label="Poll Interval (data)" value="5 000 ms" />
        <MetricRow label="Poll Interval (incidents)" value="3 000 ms" />
        <div className="divider" />
        <SectionTitle icon="🔒">Privacy</SectionTitle>
        <MetricRow label="External API calls" value="None" mono={false} />
        <MetricRow label="Audio retention"    value="In-memory only" mono={false} />
        <MetricRow label="Incident log"        value="Local SQLite" mono={false} />
      </div>
    </div>
  )
}

// ──────────────────────────────────────────────────────────────────────────────
// Main App
// ──────────────────────────────────────────────────────────────────────────────

export default function App() {
  const [active,    setActive]    = useState('Dashboard')
  const [health,    setHealth]    = useState(null)
  const [modelInfo, setModelInfo] = useState(null)
  const [result,    setResult]    = useState(null)
  const [timeline,  setTimeline]  = useState([])
  const [incidents, setIncidents] = useState([])
  const [stats,     setStats]     = useState(null)
  const [apiError,  setApiError]  = useState(false)

  const modelReady = modelInfo?.status === 'ready'

  // ── Polling ─────────────────────────────────────────────────────────────────
  const refreshData = useCallback(async () => {
    try {
      const [h, mi] = await Promise.all([api.health(), api.modelInfo()])
      setHealth(h)
      setModelInfo(mi)
      setApiError(false)
    } catch {
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
    } catch { /* network error — silently ignore incidents refresh failure */ }
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

  // ── Render active page ───────────────────────────────────────────────────────
  function renderPage() {
    switch (active) {
      case 'Dashboard':
        return <DashboardPage
          health={health} modelInfo={modelInfo}
          result={result} timeline={timeline}
          incidents={incidents} stats={stats}
          onResult={handleResult} onReload={refreshData}
        />
      case 'Analyze Voice':
        return <AnalyzeVoicePage modelInfo={modelInfo} result={result} onResult={handleResult} />
      case 'Live Monitor':
        return <LiveMonitorPage />
      case 'Incidents':
        return <IncidentsPage incidents={incidents} />
      case 'Analytics':
        return <AnalyticsPage stats={stats} timeline={timeline} />
      case 'Model Performance':
        return <ModelPerformancePage modelInfo={modelInfo} onReload={refreshData} />
      case 'Settings':
        return <SettingsPage health={health} />
      default:
        return <DashboardPage
          health={health} modelInfo={modelInfo}
          result={result} timeline={timeline}
          incidents={incidents} stats={stats}
          onResult={handleResult} onReload={refreshData}
        />
    }
  }

  return (
    <div style={{
      display: 'flex',
      minHeight: '100vh',
      background: 'radial-gradient(circle at 74% 22%, rgba(34,72,142,0.07), transparent 28%), linear-gradient(180deg, #07101c, #070d18 70%)',
    }}>
      {/* Fixed sidebar */}
      <Sidebar active={active} setActive={setActive} />

      {/* Main area — offset by sidebar width */}
      <main style={{
        marginLeft: 240,
        width: 'calc(100% - 240px)',
        display: 'flex',
        flexDirection: 'column',
        minHeight: '100vh',
      }}>
        <Topbar health={health} modelReady={modelReady} />

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
            <code style={{ marginLeft: 10, fontFamily: 'var(--font-mono)', fontSize: 'var(--fs-xs)' }}>
              uvicorn backend.app:app --port 8000
            </code>
          </div>
        )}

        {/* Page content */}
        <div style={{ flex: 1, overflowY: 'auto' }}>
          {renderPage()}
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
          background: 'rgba(5,11,19,0.6)',
        }}>
          <span>VoxShield Independent v1.0 · Voice Threat Intelligence · Local Inference Only</span>
          <span>VoxShieldNet trained from random initialization · No pretrained components · No external APIs</span>
          <span>© 2026 · SIH Demo Build</span>
        </footer>
      </main>
    </div>
  )
}
