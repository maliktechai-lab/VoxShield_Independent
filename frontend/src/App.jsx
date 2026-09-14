/**
 * VoxShield SOC Dashboard — Complete Redesign
 * Voice Threat Intelligence — Local Inference Only
 */

import { useState, useEffect, useRef, useCallback, useMemo } from 'react'
import {
  Activity, BarChart3, Bell, CheckCircle2, ChevronDown,
  Clock, Database, FileAudio, Filter, LayoutDashboard,
  Mic, Search, Settings, Shield, ShieldAlert,
  Waves, AlertTriangle, TrendingUp, Cpu, PlayCircle, X,
} from 'lucide-react'
import { api } from './api.js'

// ─────────────────────────────────────────────────────────────────────────────
// Live mic constants (mirrored from LiveDetectionOverlay)
// ─────────────────────────────────────────────────────────────────────────────

const WINDOW_SECONDS      = 4
const HOP_MS              = 2000
const BUFFER_SECONDS      = 8
const TARGET_SAMPLE_RATE  = 16000
const SCRIPT_PROC_SIZE    = 4096
const INFERENCE_TIMEOUT_MS = 5000
const N_SMOOTH            = 5
const EMA_ALPHA           = 0.35
const LEVEL_UPDATE_MS     = 100

// ─────────────────────────────────────────────────────────────────────────────
// Helpers / constants
// ─────────────────────────────────────────────────────────────────────────────

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

/**
 * Format a probability [0, 1] as a percentage with 3 decimal places.
 * Always shows the real underlying value — no artificial clamps.
 * Centralised: used by Dashboard, Analyze Voice, Live Monitor, Incidents, Analytics.
 * Examples: 0.000%  4.128%  82.341%  99.997%  100.000%
 */
function probabilityPct(v) {
  if (v === null || v === undefined) return 'N/A'
  const pct = Math.max(0, Math.min(100, Number(v) * 100))
  return pct.toFixed(3) + '%'
}

function shortTs(ts) {
  if (!ts) return '—'
  try {
    const d = new Date(ts)
    return d.toLocaleTimeString('en-US', { hour12: false })
  } catch { return ts }
}

function clamp01(v) { return Math.max(0, Math.min(1, Number(v) || 0)) }

// ─────────────────────────────────────────────────────────────────────────────
// Live mic helper functions (exact copies from LiveDetectionOverlay)
// ─────────────────────────────────────────────────────────────────────────────

function takeRecentSamples(chunks, totalSamples, sampleCount) {
  const count = Math.min(sampleCount, totalSamples)
  const out = new Float32Array(count)
  let writeOffset = count
  let remaining = count
  for (let i = chunks.length - 1; i >= 0 && remaining > 0; i--) {
    const chunk = chunks[i]
    const take = Math.min(chunk.length, remaining)
    writeOffset -= take
    out.set(chunk.subarray(chunk.length - take), writeOffset)
    remaining -= take
  }
  return out
}

function resampleLinear(input, sourceRate, targetRate) {
  if (!input.length || sourceRate === targetRate) return input
  const targetLength = Math.max(1, Math.round(input.length * targetRate / sourceRate))
  const output = new Float32Array(targetLength)
  const scale = (input.length - 1) / Math.max(1, targetLength - 1)
  for (let i = 0; i < targetLength; i++) {
    const pos = i * scale
    const left = Math.floor(pos)
    const right = Math.min(left + 1, input.length - 1)
    const frac = pos - left
    output[i] = input[left] + (input[right] - input[left]) * frac
  }
  return output
}

function encodeWavPcm16(samples, sampleRate) {
  const dataSize = samples.length * 2
  const buffer = new ArrayBuffer(44 + dataSize)
  const view = new DataView(buffer)
  const ws = (offset, s) => { for (let i = 0; i < s.length; i++) view.setUint8(offset + i, s.charCodeAt(i)) }
  ws(0, 'RIFF'); view.setUint32(4, 36 + dataSize, true)
  ws(8, 'WAVE'); ws(12, 'fmt ')
  view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true)
  view.setUint32(24, sampleRate, true); view.setUint32(28, sampleRate * 2, true)
  view.setUint16(32, 2, true); view.setUint16(34, 16, true)
  ws(36, 'data'); view.setUint32(40, dataSize, true)
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]))
    view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true)
  }
  return new Blob([buffer], { type: 'audio/wav' })
}

function makeSessionId() { return 'live_' + Math.random().toString(36).slice(2, 10) }
function makeLiveFilename() {
  const stamp = new Date().toISOString().replace(/[-:.TZ]/g, '').slice(0, 14)
  return `live_mic_${stamp}.wav`
}
function emaUpdate(prev, value, alpha) {
  if (prev === null) return value
  return alpha * value + (1 - alpha) * prev
}

// ─────────────────────────────────────────────────────────────────────────────
// Reusable primitives
// ─────────────────────────────────────────────────────────────────────────────

function StatusDot({ ok, pulse = false }) {
  return (
    <span className={pulse ? 'pulse' : ''} style={{
      display: 'inline-block', width: 8, height: 8,
      borderRadius: '50%', background: ok ? 'var(--green)' : 'var(--red)', flexShrink: 0,
    }} />
  )
}

function ProbBar({ value, color }) {
  const p = clamp01(value ?? 0) * 100
  return (
    <div className="progress-bar" style={{ flex: 1 }}>
      <div className="progress-bar-fill" style={{ width: `${p}%`, background: color }} />
    </div>
  )
}

function ThreatBadge({ level }) {
  const color = THREAT_COLORS[level] || THREAT_COLORS.UNKNOWN
  const bg    = THREAT_BG[level]    || THREAT_BG.UNKNOWN
  return (
    <span style={{
      background: bg, color, border: `1px solid ${color}40`,
      padding: '3px 10px', borderRadius: 4,
      fontSize: 'var(--fs-xxs)', fontWeight: 700,
      letterSpacing: '0.06em', textTransform: 'uppercase',
      lineHeight: 1, display: 'inline-flex', alignItems: 'center',
    }}>
      {level || 'UNKNOWN'}
    </span>
  )
}

function MetricRow({ label, value, mono = true, dim = false }) {
  return (
    <div style={{
      display: 'flex', justifyContent: 'space-between', alignItems: 'center',
      padding: '5px 0', lineHeight: 'var(--lh-compact)',
    }}>
      <span style={{ color: 'var(--text-muted)', fontSize: 'var(--fs-xs)' }}>{label}</span>
      <span style={{
        color: dim ? 'var(--text-secondary)' : 'var(--text-primary)',
        fontFamily: mono ? 'var(--font-mono)' : undefined,
        fontSize: 'var(--fs-sm)',
      }}>{value ?? '—'}</span>
    </div>
  )
}

function SectionTitle({ children, icon }) {
  return (
    <div style={{
      fontSize: 'var(--fs-xxs)', fontWeight: 600, letterSpacing: '0.08em',
      textTransform: 'uppercase', color: 'var(--text-muted)', marginBottom: 12,
      display: 'flex', alignItems: 'center', gap: 6,
    }}>
      {icon && <span style={{ fontSize: 14 }}>{icon}</span>}
      {children}
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Logo
// ─────────────────────────────────────────────────────────────────────────────

function Logo() {
  return (
    <div style={{
      height: 64, display: 'flex', gap: 12, alignItems: 'center',
      padding: '0 8px 10px', borderBottom: '1px solid var(--border)', marginBottom: 6,
    }}>
      <div style={{
        width: 43, height: 47, color: 'var(--accent)',
        display: 'grid', placeItems: 'center',
        border: '2.5px solid var(--accent)',
        clipPath: 'polygon(50% 0,96% 17%,90% 70%,50% 100%,10% 70%,4% 17%)',
        background: 'var(--accent-light)', flexShrink: 0,
      }}>
        <Waves size={20} />
      </div>
      <div>
        <div style={{ fontSize: 22, fontWeight: 800, letterSpacing: '-0.5px', color: 'var(--text-primary)', lineHeight: 1.2 }}>
          VoxShield
        </div>
        <div style={{ fontSize: 9, color: 'var(--text-muted)', letterSpacing: '1.7px', textTransform: 'uppercase', marginTop: 1 }}>
          TRUST EVERY VOICE
        </div>
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Sidebar
// ─────────────────────────────────────────────────────────────────────────────

const NAV_SECTIONS = [
  {
    label: 'OVERVIEW',
    items: [
      ['Dashboard',      LayoutDashboard],
      ['Analyze Voice',  FileAudio],
      ['Live Monitor',   Mic],
      ['Incidents',      Database],
      ['Analytics',      BarChart3],
    ],
  },
  {
    label: 'MODEL',
    items: [
      ['Model Performance', Cpu],
    ],
  },
  {
    label: 'SYSTEM',
    items: [
      ['System & Privacy', Settings],
      ['Demo Mode', PlayCircle],
    ],
  },
]

function Sidebar({ active, setActive }) {
  return (
    <aside style={{
      width: 240, minWidth: 240,
      borderRight: '1px solid var(--border)',
      background: 'var(--bg-card)',
      padding: '17px 14px 20px',
      display: 'flex', flexDirection: 'column',
      position: 'fixed', inset: '0 auto 0 0',
      overflowY: 'auto', zIndex: 100,
      boxShadow: '1px 0 0 var(--border)',
    }}>
      <Logo />
      <nav style={{ marginTop: 4 }}>
        {NAV_SECTIONS.map(({ label, items }) => (
          <div key={label}>
            <span className="nav-section-label">{label}</span>
            {items.map(([name, Icon]) => {
              const isActive = active === name
              return (
                <button
                  key={name}
                  onClick={() => setActive(name)}
                  style={{
                    width: '100%', border: 'none',
                    background: isActive ? 'var(--accent-light)' : 'transparent',
                    color: isActive ? 'var(--accent)' : 'var(--text-secondary)',
                    height: 42, borderRadius: 8,
                    display: 'flex', alignItems: 'center', gap: 11,
                    padding: '0 12px', margin: '1px 0',
                    fontSize: 'var(--fs-sm)', fontWeight: isActive ? 600 : 400,
                    textAlign: 'left', cursor: 'pointer',
                    transition: 'all 0.15s ease',
                    boxShadow: isActive ? 'inset 3px 0 0 var(--accent)' : 'none',
                  }}
                  onMouseEnter={e => { if (!isActive) e.currentTarget.style.background = 'var(--bg-hover)' }}
                  onMouseLeave={e => { if (!isActive) e.currentTarget.style.background = 'transparent' }}
                >
                  <Icon size={16} />
                  <span>{name}</span>
                </button>
              )
            })}
          </div>
        ))}
      </nav>
      <div style={{
        marginTop: 'auto',
        border: '1px solid var(--border)',
        borderRadius: 8,
        background: 'var(--accent-light)',
        padding: 14,
        display: 'flex', gap: 10, alignItems: 'flex-start',
      }}>
        <div style={{ fontSize: 22, flexShrink: 0 }}>🇮🇳</div>
        <div>
          <div style={{ fontSize: 'var(--fs-xs)', fontWeight: 700, lineHeight: 1.45, color: 'var(--accent)' }}>
            Built for a<br />Safer India
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-secondary)', lineHeight: 1.55, marginTop: 5 }}>
            Combating voice fraud<br />with AI
          </div>
        </div>
      </div>
    </aside>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Topbar
// ─────────────────────────────────────────────────────────────────────────────

function Topbar({ modelReady }) {
  const [now, setNow] = useState(new Date())
  useEffect(() => {
    const t = setInterval(() => setNow(new Date()), 1000)
    return () => clearInterval(t)
  }, [])

  return (
    <header style={{
      height: 64, borderBottom: '1px solid var(--border)',
      display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      padding: '0 28px', background: 'var(--bg-card)',
      flexShrink: 0,
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, color: 'var(--text-secondary)', fontSize: 'var(--fs-sm)' }}>
        <Shield size={16} style={{ color: 'var(--accent)' }} />
        <span>AI-Powered Voice Security for a Safer Tomorrow</span>
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 20 }}>
        <span style={{ display: 'flex', gap: 6, alignItems: 'center', color: 'var(--success)', fontSize: 'var(--fs-xs)', fontWeight: 500 }}>
          <span className="pulse" style={{ width: 7, height: 7, borderRadius: '50%', background: 'var(--success)', display: 'inline-block' }} />
          System Online
        </span>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <StatusDot ok={modelReady} pulse={!modelReady} />
          <span style={{ fontSize: 'var(--fs-xs)', color: 'var(--text-secondary)', fontWeight: 500 }}>
            {modelReady ? 'Model Ready' : 'Model Offline'}
          </span>
        </div>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: 'var(--fs-xs)', color: 'var(--text-muted)', minWidth: 68, letterSpacing: '0.04em' }}>
          {now.toLocaleTimeString('en-US', { hour12: false })}
        </div>
        <Bell size={17} style={{ color: 'var(--text-muted)', cursor: 'pointer' }} />
        <div style={{ display: 'flex', alignItems: 'center', gap: 9, cursor: 'pointer' }}>
          <div style={{
            width: 34, height: 34, borderRadius: '50%',
            display: 'grid', placeItems: 'center',
            fontSize: 14, fontWeight: 700,
            color: 'var(--accent)',
            background: 'var(--accent-light)',
            border: '1.5px solid var(--border)',
          }}>V</div>
          <div>
            <div style={{ fontSize: 'var(--fs-xs)', fontWeight: 600, color: 'var(--text-primary)', lineHeight: 1.2 }}>Admin</div>
            <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>Team Admin</div>
          </div>
          <ChevronDown size={13} style={{ color: 'var(--text-muted)' }} />
        </div>
      </div>
    </header>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// UploadPanel
// ─────────────────────────────────────────────────────────────────────────────

function UploadPanel({ onResult, modelReady }) {
  const [file, setFile]         = useState(null)
  const [dragging, setDragging] = useState(false)
  const [loading, setLoading]   = useState(false)
  const [error, setError]       = useState(null)
  const [waveData, setWaveData] = useState(null)
  const canvasRef  = useRef(null)
  const inputRef   = useRef(null)
  const dragCounter = useRef(0)

  const ACCEPTED_EXT = ['.flac', '.wav', '.ogg', '.mp3', '.m4a']

  async function drawWaveform(f) {
    try {
      const ab = await f.arrayBuffer()
      const AudioContextClass = window.AudioContext || window.webkitAudioContext
      if (!AudioContextClass) return
      const ctx = new AudioContextClass()
      const decoded = await ctx.decodeAudioData(ab)
      await ctx.close()
      const raw = decoded.getChannelData(0)
      // Downsample to 400 points
      const points = 400
      const blockSize = Math.floor(raw.length / points)
      const data = []
      for (let i = 0; i < points; i++) {
        let max = 0
        for (let j = 0; j < blockSize; j++) {
          const v = Math.abs(raw[i * blockSize + j])
          if (v > max) max = v
        }
        data.push(max)
      }
      setWaveData(data)
    } catch {
      setWaveData(null)
    }
  }

  useEffect(() => {
    if (!waveData || !canvasRef.current) return
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    const W = canvas.width, H = canvas.height
    ctx.clearRect(0, 0, W, H)
    const barW = W / waveData.length
    ctx.fillStyle = 'rgba(37,99,235,0.7)'
    const mid = H / 2
    waveData.forEach((v, i) => {
      const h = Math.max(2, v * mid)
      ctx.fillRect(i * barW, mid - h, Math.max(1, barW - 0.5), h * 2)
    })
  }, [waveData])

  function handleFile(f) {
    if (!f) return
    setError(null)
    setWaveData(null)
    setFile(f)
    drawWaveform(f)
  }

  function clearFile(e) {
    e.stopPropagation()
    setFile(null); setError(null); setWaveData(null)
    if (inputRef.current) inputRef.current.value = ''
  }

  function onDragEnter(e) { e.preventDefault(); dragCounter.current++; if (dragCounter.current === 1) setDragging(true) }
  function onDragOver(e) { e.preventDefault() }
  function onDragLeave(e) { e.preventDefault(); dragCounter.current--; if (dragCounter.current === 0) setDragging(false) }
  function onDrop(e) { e.preventDefault(); dragCounter.current = 0; setDragging(false); if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]) }

  async function analyze() {
    if (!file || loading) return
    setLoading(true); setError(null)
    try {
      const result = await api.predict(file)
      onResult({ ...result, _filename: file.name })
    } catch (e) { setError(e.message) }
    finally { setLoading(false) }
  }

  const sizeKB  = file ? (file.size / 1024).toFixed(1) : null
  const sizeMB  = file ? (file.size / (1024 * 1024)).toFixed(2) : null
  const fileExt = file ? ('.' + file.name.split('.').pop().toLowerCase()) : null
  const extOk   = fileExt ? ACCEPTED_EXT.includes(fileExt) : true

  return (
    <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
      <SectionTitle icon={<FileAudio size={14} />}>Upload Audio File</SectionTitle>

      <div
        role="button" tabIndex={0} aria-label="Click or drop audio file here"
        onClick={() => !loading && inputRef.current?.click()}
        onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); !loading && inputRef.current?.click() } }}
        onDragEnter={onDragEnter} onDragOver={onDragOver} onDragLeave={onDragLeave} onDrop={onDrop}
        style={{
          border: `2px dashed ${dragging ? 'var(--accent)' : file ? (extOk ? 'var(--green)' : 'var(--red)') : 'var(--border)'}`,
          borderRadius: 'var(--radius)', padding: '24px 16px', textAlign: 'center',
          cursor: loading ? 'not-allowed' : 'pointer',
          background: dragging ? 'var(--accent-glow)' : file ? (extOk ? 'var(--green-dim)' : 'var(--red-dim)') : 'transparent',
          transition: 'var(--transition)', userSelect: 'none', outline: 'none', position: 'relative',
        }}
      >
        <input ref={inputRef} type="file" accept={ACCEPTED_EXT.join(',')} style={{ display: 'none' }}
          onChange={e => { if (e.target.files[0]) handleFile(e.target.files[0]) }} />
        {file ? (
          <div style={{ position: 'relative' }}>
            <button aria-label="Remove selected file" onClick={clearFile} style={{
              position: 'absolute', top: -8, right: -8, width: 22, height: 22,
              borderRadius: '50%', background: 'var(--bg-card)', border: '1px solid var(--border)',
              color: 'var(--text-muted)', cursor: 'pointer', display: 'flex',
              alignItems: 'center', justifyContent: 'center', fontSize: 12, padding: 0, zIndex: 1,
            }}>✕</button>
            <div style={{ fontSize: 24, marginBottom: 6 }}>🎵</div>
            <div style={{ color: 'var(--text-primary)', fontWeight: 600, fontSize: 'var(--fs-body)', lineHeight: 'var(--lh-compact)', wordBreak: 'break-all' }}>{file.name}</div>
            <div style={{ color: 'var(--text-muted)', fontSize: 'var(--fs-xs)', marginTop: 3 }}>
              {parseFloat(sizeKB) >= 1024 ? `${sizeMB} MB` : `${sizeKB} KB`} · {file.type || fileExt || 'audio'}
            </div>
            {!extOk && <div style={{ color: 'var(--red)', fontSize: 'var(--fs-xs)', marginTop: 4, fontWeight: 500 }}>Unsupported format. Use: {ACCEPTED_EXT.join(' ')}</div>}
            <div style={{ color: 'var(--text-muted)', fontSize: 'var(--fs-xs)', marginTop: 6 }}>Click to replace</div>
          </div>
        ) : (
          <div>
            <div style={{ fontSize: 28, marginBottom: 8 }}>⬆</div>
            <div style={{ color: dragging ? 'var(--accent-hover)' : 'var(--text-secondary)', fontSize: 'var(--fs-body)', fontWeight: 500 }}>
              {dragging ? 'Drop file here' : 'Drop audio file here or click to upload'}
            </div>
            <div style={{ color: 'var(--text-muted)', fontSize: 'var(--fs-xs)', marginTop: 5 }}>
              {ACCEPTED_EXT.join(' · ')} · max 25 MB
            </div>
          </div>
        )}
      </div>

      {/* Audio Preview waveform */}
      {waveData && (
        <div>
          <div style={{ fontSize: 'var(--fs-xxs)', color: 'var(--text-muted)', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.06em' }}>Audio Preview</div>
          <canvas ref={canvasRef} width={400} height={56} style={{ width: '100%', height: 56, borderRadius: 'var(--radius-sm)', background: 'var(--bg-secondary)', display: 'block' }} />
          <div style={{ fontSize: 'var(--fs-xxs)', color: 'var(--text-muted)', marginTop: 4 }}>Waveform visualization — decorative only, not model features</div>
        </div>
      )}

      {error && (
        <div style={{ background: 'var(--red-dim)', border: '1px solid rgba(239,68,68,0.3)', color: 'var(--red)', borderRadius: 'var(--radius-sm)', padding: '9px 12px', fontSize: 'var(--fs-sm)', lineHeight: 'var(--lh-compact)', display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 8 }}>
          <span>⚠ {error}</span>
          <button onClick={() => setError(null)} style={{ background: 'none', border: 'none', color: 'var(--red)', cursor: 'pointer', fontSize: 14, padding: 0, flexShrink: 0 }} aria-label="Dismiss error">✕</button>
        </div>
      )}

      <button className="btn btn-primary" style={{ width: '100%', padding: '11px' }}
        disabled={!file || loading || !modelReady || !extOk} onClick={analyze}
        title={!modelReady ? 'Model offline' : !file ? 'Select a file first' : !extOk ? 'Unsupported format' : undefined}>
        {loading ? <><span className="spinner" /> Analyzing {file?.name}…</> : !modelReady ? '⚠ Model Offline — Start Backend' : '⚡ Analyze Audio'}
      </button>

      {!modelReady && (
        <div style={{ textAlign: 'center', color: 'var(--yellow)', fontSize: 'var(--fs-xs)', padding: '5px 8px', background: 'var(--yellow-dim)', borderRadius: 'var(--radius-sm)', fontWeight: 500 }}>
          MODEL OFFLINE — start backend: <code style={{ fontFamily: 'var(--font-mono)' }}>uvicorn backend.app:app --port 8000</code>
        </div>
      )}
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// ResultPanel
// ─────────────────────────────────────────────────────────────────────────────

function ResultPanel({ result }) {
  if (!result) {
    return (
      <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: 10, minHeight: 180 }}>
        <SectionTitle icon="📊">Detection Result</SectionTitle>
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', fontSize: 'var(--fs-sm)', padding: '20px 0', gap: 10 }}>
          <span style={{ fontSize: 32 }}>🔍</span>
          <span>Upload and analyze audio to see results</span>
        </div>
      </div>
    )
  }

  if (result.status === 'unavailable') {
    return (
      <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
        <SectionTitle icon="⚠">Detection Result</SectionTitle>
        <div style={{ background: 'var(--yellow-dim)', border: '1px solid rgba(245,158,11,0.3)', borderRadius: 'var(--radius)', padding: 20, textAlign: 'center' }}>
          <div style={{ fontSize: 28, marginBottom: 10 }}>🔌</div>
          <div style={{ color: 'var(--yellow)', fontWeight: 700, fontSize: 'var(--fs-body)' }}>MODEL OFFLINE</div>
          <div style={{ color: 'var(--text-secondary)', fontSize: 'var(--fs-sm)', marginTop: 6, lineHeight: 'var(--lh-body)' }}>{result.recommended_action}</div>
        </div>
      </div>
    )
  }

  if (result.status === 'error') {
    return (
      <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
        <SectionTitle icon="✕">Analysis Error</SectionTitle>
        <div style={{ background: 'var(--red-dim)', border: '1px solid rgba(239,68,68,0.3)', borderRadius: 'var(--radius)', padding: 16 }}>
          <div style={{ color: 'var(--red)', fontWeight: 600, fontSize: 'var(--fs-sm)' }}>⚠ {result.error}</div>
        </div>
      </div>
    )
  }

  const isSpoof   = result.classification === 'SPOOF'
  const mainColor = isSpoof ? 'var(--red)' : 'var(--green)'
  const mainBg    = isSpoof ? 'var(--red-dim)' : 'var(--green-dim)'
  const spoofP    = result.spoof_probability ?? 0
  const bfP       = result.bona_fide_probability ?? 0

  return (
    <div className="card fade-in" style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
      <SectionTitle icon="📊">Detection Result ✓</SectionTitle>

      {/* Verdict */}
      <div style={{
        background: mainBg, border: `1px solid ${mainColor}40`, borderRadius: 'var(--radius)',
        padding: '20px 16px', textAlign: 'center',
        boxShadow: isSpoof ? '0 0 24px var(--red-glow)' : '0 0 16px rgba(16,185,129,0.15)',
      }}>
        <div style={{ fontSize: 28, marginBottom: 8 }}>{isSpoof ? '🚨' : '✅'}</div>
        <div style={{ fontSize: 32, fontWeight: 800, color: mainColor, letterSpacing: '0.03em', lineHeight: 1.1 }}>
          {isSpoof ? 'SYNTHETIC SPEECH' : 'BONA FIDE'}
        </div>
        <div style={{ fontSize: 'var(--fs-sm)', color: 'var(--text-secondary)', marginTop: 6 }}>
          {isSpoof ? 'Spoof / AI-generated audio detected' : 'Genuine human speech'}
        </div>
        <div style={{ marginTop: 10 }}><ThreatBadge level={result.threat_level} /></div>
      </div>

      {/* Probabilities */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
        {[['Spoof Probability', spoofP, 'var(--red)'], ['Bona-Fide Probability', bfP, 'var(--green)']].map(([label, val, color]) => (
          <div key={label}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 5 }}>
              <span style={{ fontSize: 'var(--fs-xs)', color: 'var(--text-muted)' }}>{label}</span>
              <span style={{ fontSize: 'var(--fs-sm)', fontFamily: 'var(--font-mono)', color, fontWeight: 600 }}>{probabilityPct(val)}</span>
            </div>
            <ProbBar value={val} color={color} />
          </div>
        ))}
      </div>

      {/* Metrics */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
        {[
          ['Confidence', probabilityPct(result.confidence)],
          ['Risk Score', fmt(result.risk_score, 3)],
          ['Threshold', fmt(result.decision_threshold, 4)],
          ['Latency', `${fmt(result.latency_ms, 1)} ms`],
        ].map(([label, value]) => (
          <div key={label} style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', padding: '9px 11px' }}>
            <div style={{ fontSize: 'var(--fs-xxs)', color: 'var(--text-muted)', marginBottom: 4, textTransform: 'uppercase', letterSpacing: '0.04em' }}>{label}</div>
            <div style={{ fontSize: 'var(--fs-sm)', fontFamily: 'var(--font-mono)', color: 'var(--text-primary)', fontWeight: 500 }}>{value}</div>
          </div>
        ))}
      </div>

      {/* Recommended Action */}
      <div style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', padding: '10px 12px' }}>
        <div style={{ fontSize: 'var(--fs-xxs)', color: 'var(--text-muted)', marginBottom: 5, textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 600 }}>Recommended Action</div>
        <div style={{ fontSize: 'var(--fs-sm)', color: 'var(--text-secondary)', lineHeight: 'var(--lh-body)' }}>{result.recommended_action}</div>
      </div>

      {/* Footer meta */}
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 'var(--fs-xxs)', color: 'var(--text-muted)', flexWrap: 'wrap', gap: 4 }}>
        {result._filename && <span style={{ flex: '0 0 100%', marginBottom: 2, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>File: {result._filename}</span>}
        <span>Device: {result.device || '—'}</span>
        <span>Mode: {result.inference_mode || '—'}</span>
        <span>v{result.model_version || '—'}</span>
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// TimelinePanel
// ─────────────────────────────────────────────────────────────────────────────

function TimelinePanel({ timeline, limit = 15, onViewAll }) {
  if (!timeline || timeline.length === 0) {
    return (
      <div className="card">
        <SectionTitle icon={<Activity size={13} />}>Recent Detection Events</SectionTitle>
        <div style={{ color: 'var(--text-muted)', fontSize: 'var(--fs-sm)', padding: '14px 0', textAlign: 'center' }}>No analysis events yet</div>
      </div>
    )
  }

  return (
    <div className="card">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
        <SectionTitle icon={<Activity size={13} />}>Recent Detection Events</SectionTitle>
        {onViewAll && timeline.length > limit && (
          <button onClick={onViewAll} style={{ background: 'none', border: 'none', color: 'var(--accent)', fontSize: 'var(--fs-xs)', cursor: 'pointer', fontWeight: 600, padding: 0, display: 'flex', alignItems: 'center', gap: 4, whiteSpace: 'nowrap' }}>
            View all incidents →
          </button>
        )}
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
        {timeline.slice(0, limit).map(ev => {
          const isSpoof = ev.classification === 'SPOOF'
          const color   = THREAT_COLORS[ev.threat_level] || 'var(--text-muted)'
          return (
            <div key={ev.id} style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '6px 10px', borderRadius: 'var(--radius-sm)', background: 'var(--bg-secondary)' }}>
              <span style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', fontSize: 'var(--fs-xs)', minWidth: 68, flexShrink: 0 }}>{shortTs(ev.timestamp)}</span>
              <span style={{ color: isSpoof ? 'var(--red)' : 'var(--green)', fontWeight: 700, fontSize: 'var(--fs-xxs)', minWidth: 72, flexShrink: 0, letterSpacing: '0.03em' }}>{ev.classification}</span>
              <div style={{ flex: 1 }}>
                <div style={{ height: 3, background: 'var(--border)', borderRadius: 2, overflow: 'hidden' }}>
                  <div style={{ height: '100%', width: `${(ev.spoof_prob ?? 0) * 100}%`, background: color, borderRadius: 2 }} />
                </div>
              </div>
              <span style={{ color, fontFamily: 'var(--font-mono)', fontSize: 'var(--fs-xs)', minWidth: 60, textAlign: 'right', flexShrink: 0 }}>{ev.spoof_prob != null ? probabilityPct(ev.spoof_prob) : '—'}</span>
              <ThreatBadge level={ev.threat_level} />
            </div>
          )
        })}
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// IncidentsPanel (dashboard widget - up to 15 rows)
// ─────────────────────────────────────────────────────────────────────────────

function IncidentsPanel({ incidents }) {
  if (!incidents || incidents.length === 0) {
    return (
      <div className="card">
        <SectionTitle icon={<Database size={13} />}>Recent Incidents</SectionTitle>
        <div style={{ color: 'var(--text-muted)', fontSize: 'var(--fs-sm)', padding: '14px 0', textAlign: 'center' }}>No incidents logged yet</div>
      </div>
    )
  }

  return (
    <div className="card">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
        <SectionTitle icon={<Database size={13} />}>Recent Incidents</SectionTitle>
        <span style={{ fontSize: 'var(--fs-xxs)', color: 'var(--text-muted)' }}>{incidents.length} logged</span>
      </div>
      <div style={{ overflowX: 'auto' }}>
        <table className="data-table">
          <thead>
            <tr>{['Time','Classification','Spoof P','Risk','Threat','Latency','File'].map(h => <th key={h}>{h}</th>)}</tr>
          </thead>
          <tbody>
            {incidents.slice(0, 15).map(inc => (
              <tr key={inc.id}>
                <td style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', whiteSpace: 'nowrap', fontSize: 'var(--fs-xs)' }}>{shortTs(inc.timestamp)}</td>
                <td><span style={{ color: inc.classification === 'SPOOF' ? 'var(--red)' : 'var(--green)', fontWeight: 700, fontSize: 'var(--fs-sm)', letterSpacing: '0.03em' }}>{inc.classification}</span></td>
                <td style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)', fontSize: 'var(--fs-sm)' }}>{inc.spoof_prob != null ? probabilityPct(inc.spoof_prob) : '—'}</td>
                <td style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)', fontSize: 'var(--fs-sm)' }}>{inc.risk_score != null ? inc.risk_score.toFixed(3) : '—'}</td>
                <td><ThreatBadge level={inc.threat_level} /></td>
                <td style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', fontSize: 'var(--fs-xs)', whiteSpace: 'nowrap' }}>{inc.latency_ms != null ? `${inc.latency_ms.toFixed(0)}ms` : '—'}</td>
                <td style={{ color: 'var(--text-muted)', maxWidth: 120, fontSize: 'var(--fs-sm)' }} className="truncate">{inc.filename || '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// StatsPanel
// ─────────────────────────────────────────────────────────────────────────────

function StatsPanel({ stats }) {
  if (!stats || stats.total === 0) {
    return (
      <div className="card">
        <SectionTitle icon={<BarChart3 size={13} />}>Threat Distribution</SectionTitle>
        <div style={{ color: 'var(--text-muted)', fontSize: 'var(--fs-sm)', textAlign: 'center', padding: '14px 0' }}>No data yet</div>
      </div>
    )
  }

  const byThr = stats.by_threat || {}

  return (
    <div className="card">
      <SectionTitle icon={<BarChart3 size={13} />}>Threat Distribution</SectionTitle>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, minmax(0, 1fr))', gap: 8, marginBottom: 14 }}>
        {[
          { label: 'Total',       value: stats.total },
          { label: 'Spoof Rate',  value: stats.spoof_rate != null ? pct(stats.spoof_rate) : '—' },
          { label: 'Avg Latency', value: stats.avg_latency_ms != null ? `${Number(stats.avg_latency_ms).toFixed(2)} ms` : '—' },
        ].map(({ label, value }) => (
          <div key={label} style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', padding: '10px 6px', textAlign: 'center', minWidth: 0, overflow: 'hidden' }}>
            <div style={{ fontSize: 18, fontWeight: 700, fontFamily: 'var(--font-sans)', lineHeight: 1.2, color: 'var(--text-primary)', letterSpacing: '-0.01em', whiteSpace: 'nowrap' }}>{value}</div>
            <div style={{ fontSize: 'var(--fs-xs)', color: 'var(--text-muted)', marginTop: 4, fontWeight: 500, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{label}</div>
          </div>
        ))}
      </div>
      {Object.keys(byThr).length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          {['CRITICAL','HIGH','MEDIUM','LOW'].filter(l => byThr[l]).map(level => (
            <div key={level} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ minWidth: 72, flexShrink: 0 }}><ThreatBadge level={level} /></span>
              <div className="progress-bar" style={{ flex: 1 }}>
                <div className="progress-bar-fill" style={{ width: `${(byThr[level] / stats.total) * 100}%`, background: THREAT_COLORS[level] }} />
              </div>
              <span style={{ fontSize: 'var(--fs-sm)', fontFamily: 'var(--font-mono)', minWidth: 28, textAlign: 'right', color: 'var(--text-secondary)', flexShrink: 0 }}>{byThr[level]}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// SystemStatus
// ─────────────────────────────────────────────────────────────────────────────

function SystemStatus({ health, modelInfo }) {
  const ready = modelInfo?.status === 'ready'
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
      <SectionTitle icon={<TrendingUp size={13} />}>System Status</SectionTitle>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
        {checks.map(({ label, ok, value }) => (
          <div key={label} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '5px 0' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
              <StatusDot ok={ok} />
              <span style={{ fontSize: 'var(--fs-sm)', color: 'var(--text-secondary)' }}>{label}</span>
            </div>
            <span style={{ fontSize: 'var(--fs-sm)', fontFamily: 'var(--font-mono)', color: ok ? 'var(--text-secondary)' : 'var(--red)' }}>{value}</span>
          </div>
        ))}
      </div>
      <div className="divider" />
      <div style={{ fontSize: 'var(--fs-xs)', color: 'var(--text-muted)', lineHeight: 'var(--lh-body)', background: 'var(--bg-secondary)', borderRadius: 'var(--radius-sm)', padding: '9px 10px' }}>
        All inference runs locally. No audio data leaves this machine.
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Dashboard Page
// ─────────────────────────────────────────────────────────────────────────────

function DashboardPage({ health, modelInfo, result, timeline, incidents, stats, onResult, onReload, onNavigate }) {
  const modelReady = modelInfo?.status === 'ready'
  const lastIncident = incidents && incidents[0]

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}>
      {/* Hero header — compact */}
      <div style={{ padding: '12px 28px 8px', flexShrink: 0 }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 16, flexWrap: 'wrap' }}>
          <div>
            <h1 style={{ fontSize: 22, fontWeight: 700, lineHeight: 1.2, color: 'var(--text-primary)', letterSpacing: '-0.01em', margin: 0 }}>
              VoxShield — Voice Security Operations Center
            </h1>
            <p style={{ fontSize: 'var(--fs-sm)', color: 'var(--text-secondary)', marginTop: 2 }}>
              Detect · Analyze · Prevent
              {!modelReady && <span style={{ marginLeft: 10, color: 'var(--yellow)', fontWeight: 500 }}>⚠ Model offline.</span>}
            </p>
          </div>
          {/* Privacy badge */}
          <div style={{
            display: 'flex', alignItems: 'center', gap: 8,
            background: 'var(--green-dim)', border: '1px solid rgba(5,150,105,0.25)',
            borderRadius: 'var(--radius)', padding: '6px 12px', flexShrink: 0,
          }}>
            <span style={{ fontSize: 14 }}>🔒</span>
            <div>
              <div style={{ fontSize: 'var(--fs-xxs)', fontWeight: 700, color: 'var(--success)', letterSpacing: '0.07em', textTransform: 'uppercase' }}>LOCAL INFERENCE</div>
              <div style={{ fontSize: 10, color: 'var(--text-muted)', lineHeight: 1.4 }}>Audio processed on this machine · No external AI API</div>
            </div>
          </div>
        </div>
      </div>

      {/* KPI cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, minmax(0, 1fr))', gap: 10, padding: '0 28px 8px', flexShrink: 0 }}>
        {[
          { icon: Waves,         label: 'Total Analyses',   value: stats?.total ?? 0,                                                                     color: '#4f46e5', bg: 'rgba(79,70,229,0.10)' },
          { icon: ShieldAlert,   label: 'Threats Detected', value: stats?.by_classification?.SPOOF ?? 0,                                                  color: '#dc2626', bg: 'rgba(220,38,38,0.10)' },
          { icon: AlertTriangle, label: 'Spoof Rate',       value: stats?.spoof_rate != null ? pct(stats.spoof_rate) : '—',                               color: '#d97706', bg: 'rgba(217,119,6,0.10)' },
          { icon: Clock,         label: 'Avg Latency',      value: stats?.avg_latency_ms != null ? `${Number(stats.avg_latency_ms).toFixed(2)} ms` : '—', color: '#059669', bg: 'rgba(5,150,105,0.10)' },
        ].map(({ icon: Icon, label, value, color, bg }) => (
          <div key={label} style={{ height: 72, border: '1px solid var(--border)', borderRadius: 10, background: 'var(--bg-card)', display: 'flex', alignItems: 'center', padding: '8px 12px', gap: 10, boxShadow: 'var(--shadow-xs)', minWidth: 0, overflow: 'hidden' }}>
            <div style={{ width: 36, height: 36, borderRadius: 8, display: 'grid', placeItems: 'center', color, background: bg, flexShrink: 0 }}><Icon size={17} /></div>
            <div style={{ display: 'flex', flexDirection: 'column', flex: 1, minWidth: 0 }}>
              <span style={{ fontSize: 'var(--fs-xs)', color: 'var(--text-muted)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{label}</span>
              <strong style={{ fontSize: 'clamp(15px, 1.8vw, 22px)', fontWeight: 700, lineHeight: 1.2, color: 'var(--text-primary)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', letterSpacing: '-0.01em' }}>{value}</strong>
            </div>
          </div>
        ))}
      </div>

      {/* 3-column operational grid — fills remaining space */}
      <div style={{ display: 'grid', gridTemplateColumns: '280px 1fr 268px', gap: 10, padding: '0 28px 12px', flex: 1, minHeight: 0, overflow: 'hidden' }}>

        {/* LEFT: Live protection + Quick Actions */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8, minHeight: 0, overflow: 'auto' }}>
          {/* Live Protection */}
          <div className="card" style={{ padding: 12, flexShrink: 0 }}>
            <SectionTitle icon={<Mic size={13} />}>Live Protection</SectionTitle>
            <div style={{ marginBottom: 10 }}>
              {[
                { label: 'Microphone', value: 'Off', valueColor: 'var(--text-secondary)' },
                { label: 'Status',     value: 'Ready', valueColor: 'var(--success)' },
                ...(lastIncident ? [
                  { label: 'Last verdict',    value: lastIncident.classification, valueColor: lastIncident.classification === 'SPOOF' ? 'var(--danger)' : 'var(--success)' },
                  { label: 'Last spoof prob', value: probabilityPct(lastIncident.spoof_prob), valueColor: 'var(--text-secondary)', mono: true },
                ] : [
                  { label: 'Last verdict', value: 'No analyses yet', valueColor: 'var(--text-muted)' },
                ]),
              ].map(({ label, value, valueColor, mono }, i, arr) => (
                <div key={label} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '4px 0', borderBottom: i < arr.length - 1 ? '1px solid var(--border)' : 'none' }}>
                  <span style={{ fontSize: 'var(--fs-xs)', color: 'var(--text-muted)' }}>{label}</span>
                  <span style={{ fontSize: 'var(--fs-xs)', fontWeight: 600, color: valueColor, fontFamily: mono ? 'var(--font-mono)' : undefined }}>{value}</span>
                </div>
              ))}
            </div>
            <button className="btn btn-primary" style={{ width: '100%', padding: '7px', fontSize: 'var(--fs-xs)' }} onClick={() => onNavigate('Live Monitor')}>
              <Mic size={12} /> Open Live Monitor
            </button>
          </div>

          {/* Quick Actions */}
          <div className="card" style={{ padding: 12, flexShrink: 0 }}>
            <SectionTitle>Quick Actions</SectionTitle>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
              {[
                { label: 'Analyze Audio',     page: 'Analyze Voice',    icon: FileAudio  },
                { label: 'Start Live Monitor',page: 'Live Monitor',     icon: Mic        },
                { label: 'View Incidents',    page: 'Incidents',        icon: Database   },
                { label: 'Model Performance', page: 'Model Performance',icon: Cpu        },
              ].map(({ label, page, icon: Icon }) => (
                <button
                  key={label}
                  onClick={() => onNavigate(page)}
                  style={{
                    display: 'flex', alignItems: 'center', gap: 7,
                    background: 'var(--bg-secondary)', border: '1px solid var(--border)',
                    borderRadius: 'var(--radius-sm)', padding: '7px 9px',
                    fontSize: 'var(--fs-xs)', fontWeight: 500, color: 'var(--text-secondary)',
                    cursor: 'pointer', textAlign: 'left', transition: 'all 0.15s',
                  }}
                  onMouseEnter={e => { e.currentTarget.style.background = 'var(--accent-light)'; e.currentTarget.style.color = 'var(--accent)'; e.currentTarget.style.borderColor = 'rgba(79,70,229,0.3)' }}
                  onMouseLeave={e => { e.currentTarget.style.background = 'var(--bg-secondary)'; e.currentTarget.style.color = 'var(--text-secondary)'; e.currentTarget.style.borderColor = 'var(--border)' }}
                >
                  <Icon size={13} style={{ flexShrink: 0 }} />
                  {label}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* MIDDLE: recent detection events only */}
        <div style={{ minHeight: 0, overflow: 'auto', display: 'flex', flexDirection: 'column' }}>
          <TimelinePanel timeline={timeline} limit={6} onViewAll={() => onNavigate('Incidents')} />
        </div>

        {/* RIGHT: threat distribution + system status */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8, minHeight: 0, overflow: 'auto' }}>
          <StatsPanel stats={stats} />
          <SystemStatus health={health} modelInfo={modelInfo} />
        </div>
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Analyze Voice Page
// ─────────────────────────────────────────────────────────────────────────────

function AnalyzeVoicePage({ modelInfo, result, onResult }) {
  const modelReady = modelInfo?.status === 'ready'
  return (
    <div style={{ padding: '24px 28px' }}>
      <div className="page-header" style={{ margin: '-24px -28px 24px', padding: '24px 28px' }}>
        <h1 style={{ fontSize: 'var(--fs-h1)', fontWeight: 700, color: 'var(--text-primary)', margin: 0 }}>Analyze Voice</h1>
        <p style={{ fontSize: 'var(--fs-body)', color: 'var(--text-secondary)', marginTop: 4 }}>Upload audio to detect deepfake and cloned voices.</p>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: result ? '1fr 1fr' : '1fr', gap: 20, maxWidth: result ? 1100 : 560, transition: 'max-width 0.3s ease' }}>
        <UploadPanel onResult={onResult} modelReady={modelReady} />
        {result && <ResultPanel result={result} />}
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// LiveMonitorPage — full inline mic logic
// ─────────────────────────────────────────────────────────────────────────────

function LiveMonitorPage() {
  const [active,         setActive]         = useState(false)
  const [status,         setStatus]         = useState('idle')
  const [error,          setError]          = useState(null)
  const [level,          setLevel]          = useState(0)
  const [windowProgress, setWindowProgress] = useState(0)
  const [liveResult,     setLiveResult]     = useState(null)
  const [lastLatency,    setLastLatency]    = useState(null)
  const [smoothedProb,   setSmoothedProb]   = useState(null)
  const [windowCount,    setWindowCount]    = useState(0)
  const [recentHistory,  setRecentHistory]  = useState([])

  const streamRef         = useRef(null)
  const audioContextRef   = useRef(null)
  const sourceRef         = useRef(null)
  const processorRef      = useRef(null)
  const muteGainRef       = useRef(null)
  const timerRef          = useRef(null)
  const chunksRef         = useRef([])
  const totalSamplesRef   = useRef(0)
  const analyzingRef      = useRef(false)
  const startedAtRef      = useRef(0)
  const nextAnalysisAtRef = useRef(0)
  const lastLevelUpdateRef = useRef(0)
  const smoothedProbRef   = useRef(null)
  const sessionIdRef      = useRef(null)

  const cleanup = useCallback(async () => {
    if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null }
    try { processorRef.current?.disconnect() } catch (_) {}
    try { sourceRef.current?.disconnect() } catch (_) {}
    try { muteGainRef.current?.disconnect() } catch (_) {}
    if (streamRef.current) streamRef.current.getTracks().forEach(t => t.stop())
    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      try { await audioContextRef.current.close() } catch (_) {}
    }
    streamRef.current = null; audioContextRef.current = null; sourceRef.current = null
    processorRef.current = null; muteGainRef.current = null
    chunksRef.current = []; totalSamplesRef.current = 0; analyzingRef.current = false
    startedAtRef.current = 0; nextAnalysisAtRef.current = 0
    smoothedProbRef.current = null; sessionIdRef.current = null
    setActive(false); setStatus('idle'); setLevel(0); setWindowProgress(0); setSmoothedProb(null)
  }, [])

  const analyzeWindow = useCallback(async () => {
    if (!audioContextRef.current || analyzingRef.current) return
    const sourceRate = audioContextRef.current.sampleRate
    const requiredSamples = Math.floor(sourceRate * WINDOW_SECONDS)
    if (totalSamplesRef.current < requiredSamples) return

    analyzingRef.current = true
    setStatus('analyzing')
    try {
      const recent = takeRecentSamples(chunksRef.current, totalSamplesRef.current, requiredSamples)
      let fixed = resampleLinear(recent, sourceRate, TARGET_SAMPLE_RATE)
      const expected = TARGET_SAMPLE_RATE * WINDOW_SECONDS
      if (fixed.length > expected) { fixed = fixed.slice(0, expected) }
      else if (fixed.length < expected) { const p = new Float32Array(expected); p.set(fixed); fixed = p }

      const wav = encodeWavPcm16(fixed, TARGET_SAMPLE_RATE)
      const wavFile = new File([wav], makeLiveFilename(), { type: 'audio/wav' })

      let result
      const controller = new AbortController()
      const tid = setTimeout(() => controller.abort(), INFERENCE_TIMEOUT_MS)
      try { result = await api.predictLive(wavFile, sessionIdRef.current) }
      finally { clearTimeout(tid) }

      if (!result || typeof result.spoof_probability !== 'number') throw new Error('Malformed response')

      const rawProb = clamp01(result.spoof_probability)
      const newSmoothed = emaUpdate(smoothedProbRef.current, rawProb, EMA_ALPHA)
      smoothedProbRef.current = newSmoothed
      setWindowCount(c => c + 1)
      setRecentHistory(prev => [...prev, rawProb].slice(-N_SMOOTH))
      setSmoothedProb(newSmoothed)
      setLiveResult(result)
      setLastLatency(result.latency_ms ?? null)
      setError(null)
    } catch (e) {
      setError(e.name === 'AbortError' ? 'Inference timeout (>5s)' : e.message || 'Unknown error')
    } finally {
      analyzingRef.current = false
      if (streamRef.current && audioContextRef.current) setStatus('listening')
    }
  }, [])

  const start = useCallback(async () => {
    if (active) return
    setError(null); setLiveResult(null); setLastLatency(null)
    setSmoothedProb(null); setWindowCount(0); setRecentHistory([])
    setStatus('requesting')

    if (!navigator.mediaDevices?.getUserMedia) {
      setError('getUserMedia not supported.'); setStatus('error'); return
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false } })
      const AudioContextClass = window.AudioContext || window.webkitAudioContext
      if (!AudioContextClass) throw new Error('Web Audio API unavailable.')
      const context = new AudioContextClass()
      if (context.state === 'suspended') await context.resume()
      if (context.state !== 'running') throw new Error('AudioContext failed to start.')

      const source = context.createMediaStreamSource(stream)
      const processor = context.createScriptProcessor(SCRIPT_PROC_SIZE, 1, 1)
      const muteGain = context.createGain(); muteGain.gain.value = 0

      processor.onaudioprocess = event => {
        const input = event.inputBuffer.getChannelData(0)
        const copy = new Float32Array(input)
        chunksRef.current.push(copy)
        totalSamplesRef.current += copy.length
        const maxSamples = Math.floor(context.sampleRate * BUFFER_SECONDS)
        while (totalSamplesRef.current > maxSamples && chunksRef.current.length > 1) {
          const removed = chunksRef.current.shift()
          totalSamplesRef.current -= removed.length
        }
        const now = performance.now()
        if (now - lastLevelUpdateRef.current > LEVEL_UPDATE_MS) {
          lastLevelUpdateRef.current = now
          let ss = 0
          for (let i = 0; i < input.length; i++) ss += input[i] * input[i]
          setLevel(clamp01(Math.sqrt(ss / Math.max(1, input.length)) * 8))
        }
      }

      source.connect(processor); processor.connect(muteGain); muteGain.connect(context.destination)
      streamRef.current = stream; audioContextRef.current = context
      sourceRef.current = source; processorRef.current = processor; muteGainRef.current = muteGain
      startedAtRef.current = Date.now(); nextAnalysisAtRef.current = WINDOW_SECONDS * 1000
      sessionIdRef.current = makeSessionId()
      setActive(true); setStatus('listening')

      timerRef.current = window.setInterval(() => {
        const elapsed = Date.now() - startedAtRef.current
        setWindowProgress(Math.min(1, elapsed / (WINDOW_SECONDS * 1000)))
        if (elapsed >= nextAnalysisAtRef.current) {
          nextAnalysisAtRef.current += HOP_MS
          void analyzeWindow()
        }
      }, 250)
    } catch (e) {
      const msg = e.name === 'NotAllowedError' ? 'Microphone permission denied.' : e.name === 'NotFoundError' ? 'No microphone found.' : e.message || 'Failed to start.'
      setError(msg); setStatus('error'); await cleanup()
    }
  }, [active, analyzeWindow, cleanup])

  useEffect(() => () => { void cleanup() }, [cleanup])

  const rawProb = liveResult ? clamp01(liveResult.spoof_probability ?? 0) : null
  const modelThreshold = liveResult?.decision_threshold ?? 0.6729
  const smoothedIsSpoof = smoothedProb !== null ? smoothedProb >= modelThreshold : rawProb !== null ? rawProb >= modelThreshold : null
  const verdictText  = smoothedIsSpoof === null ? null : smoothedIsSpoof ? 'SPOOF DETECTED' : 'BONA FIDE'
  const verdictColor = smoothedIsSpoof === null ? 'var(--text-muted)' : smoothedIsSpoof ? 'var(--red)' : 'var(--green)'
  const verdictBg    = smoothedIsSpoof === null ? 'transparent' : smoothedIsSpoof ? 'rgba(239,68,68,0.06)' : 'rgba(16,185,129,0.06)'
  const analyzing = status === 'analyzing'
  const hasResult = liveResult !== null && liveResult.classification !== undefined

  return (
    <div style={{ padding: '24px 28px' }}>
      <div className="page-header" style={{ margin: '-24px -28px 24px', padding: '24px 28px' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 12 }}>
          <div>
            <h1 style={{ fontSize: 'var(--fs-hero)', fontWeight: 700, color: 'var(--text-primary)', margin: 0, letterSpacing: '-0.01em' }}>Live Voice Monitor</h1>
            <p style={{ fontSize: 'var(--fs-body)', color: 'var(--text-secondary)', marginTop: 4 }}>
              Real-time microphone analysis · 4-second rolling windows · VoxShieldNet local inference
            </p>
          </div>
          {/* Status badges */}
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
            {[
              { label: active ? '● LIVE' : '○ OFFLINE', color: active ? 'var(--danger)' : 'var(--text-muted)', bg: active ? 'var(--danger-bg)' : 'var(--bg-card-muted)', border: active ? 'var(--danger-border)' : 'var(--border)', pulse: active },
              { label: active ? 'MIC ACTIVE' : 'MIC OFF', color: active ? 'var(--success)' : 'var(--text-muted)', bg: active ? 'var(--success-bg)' : 'var(--bg-card-muted)', border: active ? 'var(--success-border)' : 'var(--border)', pulse: false },
              ...(analyzing ? [{ label: 'ANALYZING', color: 'var(--accent)', bg: 'var(--accent-light)', border: 'rgba(79,70,229,0.3)', pulse: true }] : []),
            ].map(({ label, color, bg, border, pulse }) => (
              <span key={label} style={{ display: 'inline-flex', alignItems: 'center', gap: 6, fontSize: 'var(--fs-xxs)', fontWeight: 700, letterSpacing: '0.07em', padding: '5px 12px', borderRadius: 'var(--radius-sm)', color, background: bg, border: `1px solid ${border}` }}>
                <span className={pulse ? 'pulse' : ''} style={{ width: 6, height: 6, borderRadius: '50%', background: color, flexShrink: 0, display: 'inline-block' }} />
                {label}
              </span>
            ))}
          </div>
        </div>
      </div>

      <div style={{ maxWidth: 820 }}>

        {/* Hero verdict — always present, shows idle state or live result */}
        <div className={hasResult && verdictText ? 'fade-in' : ''} style={{
          background: hasResult && verdictText ? verdictBg : 'var(--bg-card)',
          border: `1px solid ${hasResult && verdictText ? verdictColor + '30' : 'var(--border)'}`,
          borderRadius: 'var(--radius-xl)', padding: '32px', textAlign: 'center', marginBottom: 20,
          boxShadow: hasResult && verdictText && smoothedIsSpoof ? '0 0 40px rgba(220,38,38,0.08)' : hasResult && verdictText ? '0 0 40px rgba(5,150,105,0.08)' : 'var(--shadow-xs)',
          transition: 'all 0.3s ease',
        }}>
          {hasResult && verdictText ? (
            <>
              <div className="live-page-verdict" style={{ color: verdictColor }}>
                {verdictText}
              </div>
              <div style={{ fontSize: 'var(--fs-h2)', fontFamily: 'var(--font-mono)', fontWeight: 700, color: verdictColor, opacity: 0.9, marginBottom: 10, marginTop: 6 }}>
                {probabilityPct(rawProb)} spoof
              </div>
              <div style={{ marginTop: 4 }}>
                <ThreatBadge level={liveResult?.threat_level} />
              </div>
              <div style={{ fontSize: 'var(--fs-xs)', color: 'var(--text-muted)', marginTop: 12 }}>
                Smoothed verdict · EMA α={EMA_ALPHA} · last {N_SMOOTH} windows
              </div>
            </>
          ) : (
            <>
              <div style={{ fontSize: 48, marginBottom: 12, opacity: 0.4 }}>🎙</div>
              <div style={{ fontSize: 28, fontWeight: 800, color: 'var(--text-secondary)', marginBottom: 6, letterSpacing: '-0.01em' }}>
                {active ? 'Warming up…' : 'MICROPHONE READY'}
              </div>
              <div style={{ fontSize: 'var(--fs-body)', color: 'var(--text-muted)', marginBottom: 4 }}>
                {active ? `First verdict in ~${WINDOW_SECONDS}s` : 'Ready to analyze live voice input'}
              </div>
              {!active && (
                <div style={{ fontSize: 'var(--fs-xs)', color: 'var(--text-muted)', marginTop: 4 }}>
                  4-second rolling windows · EMA smoothing · Local inference
                </div>
              )}
            </>
          )}
        </div>

        {/* Audio Activity + metrics */}
        <div style={{ marginBottom: 20 }}>
          <div style={{ fontSize: 'var(--fs-xxs)', fontWeight: 600, letterSpacing: '0.08em', textTransform: 'uppercase', color: 'var(--text-muted)', marginBottom: 10 }}>Audio Activity</div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12 }}>
            {[
              { label: 'Audio Level', value: active ? `${Math.round(level * 100)}%` : '—', bar: active ? level : null, barColor: 'var(--accent)' },
              { label: 'Window', value: active ? `${Math.round(windowProgress * WINDOW_SECONDS)} / ${WINDOW_SECONDS}s` : '—', bar: active ? windowProgress : null, barColor: 'var(--success)' },
              { label: 'Analyses', value: windowCount || '0', bar: null },
              { label: 'Latency', value: lastLatency != null ? `${Number(lastLatency).toFixed(0)} ms` : '—', bar: null },
            ].map(({ label, value, bar, barColor }) => (
              <div key={label} style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 'var(--radius)', padding: '12px 14px', boxShadow: 'var(--shadow-xs)' }}>
                <div style={{ fontSize: 'var(--fs-xxs)', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 6 }}>{label}</div>
                <div style={{ fontSize: 20, fontFamily: 'var(--font-mono)', fontWeight: 700, color: 'var(--text-primary)', marginBottom: bar != null ? 8 : 0 }}>{value}</div>
                {bar != null && <div style={{ height: 4, background: 'var(--bg-hover)', borderRadius: 2, overflow: 'hidden' }}><div style={{ width: `${bar * 100}%`, height: '100%', background: barColor, transition: 'width 0.2s', borderRadius: 2 }} /></div>}
              </div>
            ))}
          </div>
        </div>

        {/* Probability section */}
        {hasResult && (
          <div className="card fade-in" style={{ marginBottom: 16 }}>
            <SectionTitle>Raw Probability Analysis</SectionTitle>
            <div style={{ marginBottom: 14 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 6 }}>
                <span style={{ fontSize: 'var(--fs-xs)', color: 'var(--text-muted)' }}>Raw Spoof Probability (this window)</span>
                <span style={{ fontSize: 24, fontFamily: 'var(--font-mono)', fontWeight: 700, color: verdictColor }}>{probabilityPct(rawProb)}</span>
              </div>
              <div style={{ height: 8, background: 'var(--bg-hover)', borderRadius: 4, overflow: 'hidden' }}>
                <div style={{ width: `${clamp01(rawProb ?? 0) * 100}%`, height: '100%', background: verdictColor, transition: 'width 0.3s ease', borderRadius: 4 }} />
              </div>
            </div>
            <div style={{ marginBottom: 14 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 6 }}>
                <span style={{ fontSize: 'var(--fs-xs)', color: 'var(--text-muted)' }}>Smoothed Probability (EMA α={EMA_ALPHA})</span>
                <span style={{ fontSize: 'var(--fs-body)', fontFamily: 'var(--font-mono)', fontWeight: 600, color: 'var(--text-secondary)' }}>{probabilityPct(smoothedProb)}</span>
              </div>
              <div style={{ height: 6, background: 'var(--bg-hover)', borderRadius: 3, overflow: 'hidden', position: 'relative' }}>
                <div style={{ width: `${clamp01(smoothedProb ?? 0) * 100}%`, height: '100%', background: 'var(--accent)', transition: 'width 0.3s ease', borderRadius: 3 }} />
                <div style={{ position: 'absolute', top: 0, left: `${modelThreshold * 100}%`, width: 2, height: '100%', background: 'var(--warning)', borderRadius: 1 }} />
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 'var(--fs-xxs)', color: 'var(--text-muted)', marginTop: 4 }}>
                <span>Threshold: {modelThreshold.toFixed(4)}</span>
                <span style={{ color: 'var(--warning)' }}>▲ threshold marker</span>
              </div>
            </div>

            {/* Sparkbar history */}
            {recentHistory.length > 0 && (
              <div>
                <div style={{ fontSize: 'var(--fs-xxs)', fontWeight: 600, letterSpacing: '0.08em', textTransform: 'uppercase', color: 'var(--text-muted)', marginBottom: 8 }}>
                  Recent Windows — last {recentHistory.length}
                </div>
                <div className="prob-sparkbar">
                  {recentHistory.map((p, i) => {
                    const h = Math.max(4, Math.round(clamp01(p) * 32))
                    const c = p >= modelThreshold ? 'var(--danger)' : 'var(--success)'
                    return <div key={i} style={{ flex: 1, height: h, background: c, borderRadius: 2, opacity: 0.55 + 0.45 * (i / recentHistory.length) }} title={probabilityPct(p)} />
                  })}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Error */}
        {error && (
          <div style={{ background: 'var(--danger-bg)', border: '1px solid var(--danger-border)', color: 'var(--danger)', borderRadius: 'var(--radius-sm)', padding: '10px 14px', fontSize: 'var(--fs-sm)', marginBottom: 16 }}>⚠ {error}</div>
        )}

        {/* Status message */}
        <div style={{ fontSize: 'var(--fs-sm)', color: 'var(--text-muted)', marginBottom: 16, minHeight: 20, display: 'flex', alignItems: 'center', gap: 6 }}>
          {analyzing ? <><span className="spinner" /> Analyzing latest 4-second window…</>
            : active && windowCount === 0 ? `🎙 Warming up — first verdict in ~${WINDOW_SECONDS}s…`
            : active ? '🎙 Listening — verdict updates every ~2s'
            : 'Microphone off. Click Start to begin real-time detection.'}
        </div>

        {/* Start/Stop button */}
        <button
          className={active ? 'btn btn-danger' : 'btn btn-primary'}
          style={{ padding: '14px 40px', fontSize: 'var(--fs-body)', fontWeight: 700, marginBottom: 24, minWidth: 220 }}
          onClick={() => { if (active) void cleanup(); else void start() }}
          disabled={status === 'requesting'}>
          {status === 'requesting' ? <><span className="spinner" /> Requesting microphone…</> : active ? '■ Stop Live Detection' : '🎙 Start Live Detection'}
        </button>

        {/* Detection scope */}
        <div className="scope-card">
          <div style={{ fontSize: 'var(--fs-xs)', fontWeight: 700, color: 'var(--yellow)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.06em' }}>Detection Scope</div>
          <div style={{ fontSize: 'var(--fs-xs)', color: 'var(--text-secondary)', lineHeight: 'var(--lh-body)' }}>
            <div style={{ color: 'var(--green)', marginBottom: 4 }}>✓ Detects: direct digital TTS, voice-conversion files (logical-access attacks, ASVspoof5 types)</div>
            <div style={{ color: 'var(--yellow)' }}>⚠ Limitation: acoustic replay (TTS via speaker re-recorded by mic) — not reliably detected. Model trained on ASVspoof5 logical-access only.</div>
          </div>
        </div>
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// IncidentsPage — full page with filters and expandable rows
// ─────────────────────────────────────────────────────────────────────────────

function IncidentsPage({ incidents }) {
  const [search, setSearch]           = useState('')
  const [classFilter, setClassFilter] = useState('All')
  const [threatFilter, setThreatFilter] = useState('All')
  const [sourceFilter, setSourceFilter] = useState('All')
  const [expandedId, setExpandedId]   = useState(null)

  const filtered = useMemo(() => {
    if (!incidents) return []
    return incidents.filter(inc => {
      if (search && !((inc.filename || '').toLowerCase().includes(search.toLowerCase()) || (inc.session_id || '').toLowerCase().includes(search.toLowerCase()))) return false
      if (classFilter !== 'All' && inc.classification !== classFilter) return false
      if (threatFilter !== 'All' && inc.threat_level !== threatFilter) return false
      if (sourceFilter === 'Live Sessions' && !(inc.filename || '').startsWith('live_mic_')) return false
      if (sourceFilter === 'Uploaded Files' && (inc.filename || '').startsWith('live_mic_')) return false
      return true
    })
  }, [incidents, search, classFilter, threatFilter, sourceFilter])

  return (
    <div style={{ padding: '24px 28px' }}>
      <div className="page-header" style={{ margin: '-24px -28px 24px', padding: '24px 28px' }}>
        <h1 style={{ fontSize: 'var(--fs-h1)', fontWeight: 700, color: 'var(--text-primary)', margin: 0 }}>Incidents</h1>
        <p style={{ fontSize: 'var(--fs-body)', color: 'var(--text-secondary)', marginTop: 4 }}>Full incident log from all analyses.</p>
      </div>

      {/* Filter toolbar */}
      <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', marginBottom: 12, alignItems: 'center' }}>
        <div style={{ position: 'relative', flex: '1 1 200px', minWidth: 160 }}>
          <Search size={14} style={{ position: 'absolute', left: 9, top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)' }} />
          <input className="filter-input" placeholder="Search filename / session…" value={search} onChange={e => setSearch(e.target.value)} style={{ paddingLeft: 30, width: '100%' }} />
        </div>
        <select className="filter-select" value={classFilter} onChange={e => setClassFilter(e.target.value)}>
          <option value="All">All Classifications</option>
          <option value="BONA_FIDE">BONA_FIDE</option>
          <option value="SPOOF">SPOOF</option>
        </select>
        <select className="filter-select" value={threatFilter} onChange={e => setThreatFilter(e.target.value)}>
          <option value="All">All Threats</option>
          <option value="CRITICAL">CRITICAL</option>
          <option value="HIGH">HIGH</option>
          <option value="MEDIUM">MEDIUM</option>
          <option value="LOW">LOW</option>
        </select>
        <select className="filter-select" value={sourceFilter} onChange={e => setSourceFilter(e.target.value)}>
          <option value="All">All Sources</option>
          <option value="Live Sessions">Live Sessions</option>
          <option value="Uploaded Files">Uploaded Files</option>
        </select>
        {(search || classFilter !== 'All' || threatFilter !== 'All' || sourceFilter !== 'All') && (
          <button onClick={() => { setSearch(''); setClassFilter('All'); setThreatFilter('All'); setSourceFilter('All') }} style={{ background: 'none', border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', color: 'var(--text-muted)', cursor: 'pointer', padding: '7px 10px', display: 'flex', alignItems: 'center', gap: 4, fontSize: 'var(--fs-xs)' }}>
            <X size={12} /> Clear
          </button>
        )}
      </div>

      <div style={{ fontSize: 'var(--fs-xs)', color: 'var(--text-muted)', marginBottom: 12 }}>
        Showing {filtered.length} of {incidents?.length ?? 0} incidents
      </div>

      {filtered.length === 0 ? (
        <div className="card" style={{ textAlign: 'center', padding: '32px', color: 'var(--text-muted)' }}>
          <div style={{ fontSize: 28, marginBottom: 8 }}>🗂</div>
          <div>{incidents?.length === 0 ? 'No incidents logged yet.' : 'No incidents match the current filters.'}</div>
        </div>
      ) : (
        <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
          <div style={{ overflowX: 'auto' }}>
            <table className="data-table" style={{ minWidth: 700 }}>
              <thead>
                <tr>{['Time','Classification','Spoof Prob','Risk Score','Threat','Latency','Source'].map(h => <th key={h}>{h}</th>)}</tr>
              </thead>
              <tbody>
                {filtered.map(inc => {
                  const isExpanded = expandedId === inc.id
                  return [
                    <tr key={inc.id} style={{ cursor: 'pointer' }} onClick={() => setExpandedId(isExpanded ? null : inc.id)}>
                      <td style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', whiteSpace: 'nowrap', fontSize: 'var(--fs-xs)' }}>{shortTs(inc.timestamp)}</td>
                      <td><span style={{ color: inc.classification === 'SPOOF' ? 'var(--red)' : 'var(--green)', fontWeight: 700, fontSize: 'var(--fs-sm)' }}>{inc.classification}</span></td>
                      <td style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)', fontSize: 'var(--fs-sm)' }}>{inc.spoof_prob != null ? probabilityPct(inc.spoof_prob) : '—'}</td>
                      <td style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)', fontSize: 'var(--fs-sm)' }}>{inc.risk_score != null ? inc.risk_score.toFixed(3) : '—'}</td>
                      <td><ThreatBadge level={inc.threat_level} /></td>
                      <td style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', fontSize: 'var(--fs-xs)', whiteSpace: 'nowrap' }}>{inc.latency_ms != null ? `${inc.latency_ms.toFixed(0)}ms` : '—'}</td>
                      <td style={{ color: 'var(--text-muted)', maxWidth: 160, fontSize: 'var(--fs-sm)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{inc.filename || '—'}</td>
                    </tr>,
                    isExpanded && (
                      <tr key={`${inc.id}-expanded`} className="incident-row-expanded">
                        <td colSpan={7} style={{ padding: '12px 16px' }}>
                          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))', gap: 8 }}>
                            {[
                              ['Session ID', inc.session_id],
                              ['File / Source', inc.filename],
                              ['Device', inc.device],
                              ['Inference Mode', inc.inference_mode],
                              ['Model Version', inc.model_version],
                              ['Recommended Action', inc.recommended_action],
                              ['Timestamp', inc.timestamp],
                              ['ID', inc.id],
                            ].filter(([, v]) => v != null).map(([label, value]) => (
                              <div key={label}>
                                <div style={{ fontSize: 'var(--fs-xxs)', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 2 }}>{label}</div>
                                <div style={{ fontSize: 'var(--fs-xs)', color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)', wordBreak: 'break-all' }}>{String(value)}</div>
                              </div>
                            ))}
                          </div>
                        </td>
                      </tr>
                    ),
                  ]
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// AnalyticsPage
// ─────────────────────────────────────────────────────────────────────────────

function AnalyticsPage({ stats, timeline }) {
  const byThr = stats?.by_threat || {}
  const byClass = stats?.by_classification || {}
  const total = stats?.total ?? 0

  // SVG sparkline from timeline
  const svgWidth  = 600
  const svgHeight = 80
  const pts = useMemo(() => {
    if (!timeline || timeline.length < 2) return ''
    const valid = timeline.filter(e => e.spoof_prob != null)
    if (valid.length < 2) return ''
    const xStep = svgWidth / (valid.length - 1)
    return valid.map((e, i) => {
      const x = i * xStep
      const y = svgHeight - clamp01(e.spoof_prob) * (svgHeight - 8) - 4
      return `${x},${y}`
    }).join(' ')
  }, [timeline])

  return (
    <div style={{ padding: '24px 28px' }}>
      <div className="page-header" style={{ margin: '-24px -28px 24px', padding: '24px 28px' }}>
        <h1 style={{ fontSize: 'var(--fs-h1)', fontWeight: 700, color: 'var(--text-primary)', margin: 0 }}>Analytics</h1>
        <p style={{ fontSize: 'var(--fs-body)', color: 'var(--text-secondary)', marginTop: 4 }}>Aggregate statistics and threat distribution.</p>
      </div>

      {/* KPI row */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, minmax(0, 1fr))', gap: 14, marginBottom: 20 }}>
        {[
          { label: 'Total Analyses',  value: total, color: 'var(--accent)' },
          { label: 'Spoof Rate',      value: stats?.spoof_rate != null ? pct(stats.spoof_rate) : '—', color: 'var(--red)' },
          { label: 'Avg Latency',     value: stats?.avg_latency_ms != null ? `${Number(stats.avg_latency_ms).toFixed(2)} ms` : '—', color: 'var(--green)' },
          { label: 'Peak Threat',     value: byThr.CRITICAL ? 'CRITICAL' : byThr.HIGH ? 'HIGH' : byThr.MEDIUM ? 'MEDIUM' : byThr.LOW ? 'LOW' : '—', color: THREAT_COLORS[byThr.CRITICAL ? 'CRITICAL' : byThr.HIGH ? 'HIGH' : 'MEDIUM'] || 'var(--text-muted)' },
        ].map(({ label, value, color }) => (
          <div key={label} style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 10, padding: '16px', boxShadow: 'var(--shadow)', minWidth: 0, overflow: 'hidden' }}>
            <div style={{ fontSize: 'var(--fs-xs)', color: 'var(--text-muted)', marginBottom: 6, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{label}</div>
            <div className="kpi-value" style={{ color }}>{value}</div>
          </div>
        ))}
      </div>

      {/* Two-column: threat distribution + classification split */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: 16, marginBottom: 20 }}>
        {/* Threat distribution */}
        <div className="card">
          <SectionTitle icon={<BarChart3 size={13} />}>Threat Distribution</SectionTitle>
          {total === 0 ? <div style={{ color: 'var(--text-muted)', fontSize: 'var(--fs-sm)', textAlign: 'center', padding: '20px 0' }}>No data yet</div> : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              {['CRITICAL','HIGH','MEDIUM','LOW'].map(level => {
                const count = byThr[level] ?? 0
                const pctVal = total > 0 ? ((count / total) * 100).toFixed(1) : '0.0'
                return (
                  <div key={level}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4, alignItems: 'center' }}>
                      <ThreatBadge level={level} />
                      <span style={{ fontSize: 'var(--fs-xs)', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>{count} ({pctVal}%)</span>
                    </div>
                    <div style={{ height: 6, background: 'var(--bg-secondary)', borderRadius: 3, overflow: 'hidden' }}>
                      <div style={{ width: `${(count / Math.max(1, total)) * 100}%`, height: '100%', background: THREAT_COLORS[level], borderRadius: 3 }} />
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </div>

        {/* Classification split */}
        <div className="card">
          <SectionTitle icon={<CheckCircle2 size={13} />}>Classification Split</SectionTitle>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 16 }}>
            <div style={{ background: 'var(--green-dim)', border: '1px solid rgba(16,185,129,0.25)', borderRadius: 'var(--radius)', padding: '16px', textAlign: 'center' }}>
              <div style={{ fontSize: 36, fontWeight: 800, fontFamily: 'var(--font-mono)', color: 'var(--green)', lineHeight: 1.1 }}>{byClass.BONA_FIDE ?? 0}</div>
              <div style={{ fontSize: 'var(--fs-xs)', color: 'var(--green)', marginTop: 4, fontWeight: 600 }}>BONA FIDE</div>
            </div>
            <div style={{ background: 'var(--red-dim)', border: '1px solid rgba(239,68,68,0.25)', borderRadius: 'var(--radius)', padding: '16px', textAlign: 'center' }}>
              <div style={{ fontSize: 36, fontWeight: 800, fontFamily: 'var(--font-mono)', color: 'var(--red)', lineHeight: 1.1 }}>{byClass.SPOOF ?? 0}</div>
              <div style={{ fontSize: 'var(--fs-xs)', color: 'var(--red)', marginTop: 4, fontWeight: 600 }}>SPOOF</div>
            </div>
          </div>
          <div style={{ textAlign: 'center', fontSize: 'var(--fs-sm)', color: 'var(--text-muted)' }}>
            Spoof Rate: <strong style={{ color: 'var(--text-secondary)' }}>{stats?.spoof_rate != null ? pct(stats.spoof_rate) : '—'}</strong>
          </div>
        </div>
      </div>

      {/* Timeline sparkline */}
      <div className="card" style={{ marginBottom: 20 }}>
        <SectionTitle icon={<TrendingUp size={13} />}>Detection Timeline (Recent Events)</SectionTitle>
        {(!timeline || timeline.length < 2) ? (
          <div style={{ color: 'var(--text-muted)', fontSize: 'var(--fs-sm)', textAlign: 'center', padding: '20px 0' }}>Need at least 2 events for chart</div>
        ) : (
          <div>
            <svg viewBox={`0 0 ${svgWidth} ${svgHeight}`} style={{ width: '100%', height: 80, display: 'block' }} preserveAspectRatio="none">
              <line x1="0" y1={svgHeight - 4} x2={svgWidth} y2={svgHeight - 4} stroke="var(--border)" strokeWidth="1" />
              <line x1="0" y1={(svgHeight - 8) * 0.5 + 4} x2={svgWidth} y2={(svgHeight - 8) * 0.5 + 4} stroke="rgba(100,116,139,0.3)" strokeWidth="1" strokeDasharray="4,4" />
              {pts && <polyline points={pts} fill="none" stroke="var(--accent)" strokeWidth="2" opacity="0.8" />}
              {timeline.filter(e => e.spoof_prob != null).map((e, i) => {
                const valid = timeline.filter(ev => ev.spoof_prob != null)
                const xStep = svgWidth / Math.max(1, valid.length - 1)
                const x = i * xStep
                const y = svgHeight - clamp01(e.spoof_prob) * (svgHeight - 8) - 4
                const c = e.classification === 'SPOOF' ? 'var(--red)' : 'var(--green)'
                return <circle key={e.id} cx={x} cy={y} r="3" fill={c} />
              })}
            </svg>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 'var(--fs-xxs)', color: 'var(--text-muted)', marginTop: 4 }}>
              <span>Older</span><span>↑ Spoof Probability</span><span>Newer</span>
            </div>
          </div>
        )}
      </div>

      {/* Latency card */}
      <div className="card">
        <SectionTitle icon={<Clock size={13} />}>Inference Latency</SectionTitle>
        <div style={{ display: 'flex', alignItems: 'center', gap: 20 }}>
          <div className="kpi-value" style={{ color: 'var(--green)' }}>{stats?.avg_latency_ms != null ? `${Number(stats.avg_latency_ms).toFixed(2)} ms` : '—'}</div>
          <div style={{ fontSize: 'var(--fs-sm)', color: 'var(--text-muted)' }}>Average inference time per 4-second audio window</div>
        </div>
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// ModelPerformancePage
// ─────────────────────────────────────────────────────────────────────────────

function ModelPerformancePage({ modelInfo, onReload }) {
  const [reloading, setReloading] = useState(false)
  const ready    = modelInfo?.status === 'ready'
  const vm       = modelInfo?.val_metrics || {}
  const dsType   = modelInfo?.dataset_type || '—'

  async function handleReload() {
    setReloading(true)
    try { await api.reloadModel(); onReload() } catch (_) {}
    finally { setReloading(false) }
  }

  return (
    <div style={{ padding: '24px 28px' }}>
      <div className="page-header" style={{ margin: '-24px -28px 24px', padding: '24px 28px' }}>
        <h1 style={{ fontSize: 'var(--fs-h1)', fontWeight: 700, color: 'var(--text-primary)', margin: 0 }}>Model Performance</h1>
        <p style={{ fontSize: 'var(--fs-body)', color: 'var(--text-secondary)', marginTop: 4 }}>VoxShieldNet architecture, validation metrics, and risk engine.</p>
      </div>

      <div style={{ maxWidth: 860 }}>
        {/* Section 1: Model identity */}
        <div className="card" style={{ marginBottom: 16 }}>
          <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 12, marginBottom: 14 }}>
            <div>
              <h2 style={{ fontSize: 22, fontWeight: 800, color: 'var(--text-primary)', margin: 0 }}>VoxShieldNet</h2>
              <div style={{ fontSize: 'var(--fs-sm)', color: 'var(--text-muted)', marginTop: 4 }}>Custom neural network trained from random initialization</div>
            </div>
            <button className="btn btn-secondary" style={{ padding: '6px 12px', fontSize: 'var(--fs-xs)' }} onClick={handleReload} disabled={reloading}>
              {reloading ? '...' : '↺ Reload'}
            </button>
          </div>
          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 14 }}>
            {['ASVspoof5', 'LOCAL ONLY', 'pretrained: NO', 'external APIs: NONE'].map(b => (
              <span key={b} style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)', borderRadius: 4, padding: '3px 10px', fontSize: 'var(--fs-xxs)', fontWeight: 600, color: 'var(--text-secondary)', letterSpacing: '0.04em' }}>{b}</span>
            ))}
            {ready && <span style={{ background: 'var(--green-dim)', border: '1px solid rgba(16,185,129,0.3)', borderRadius: 4, padding: '3px 10px', fontSize: 'var(--fs-xxs)', fontWeight: 600, color: 'var(--green)' }}>READY</span>}
          </div>
          {ready && (
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2px 24px' }}>
              {[
                ['Parameters', (modelInfo.parameter_count ?? 3833218).toLocaleString()],
                ['Version',    modelInfo.model_version],
                ['Device',     modelInfo.device],
                ['Dataset',    dsType],
                ['Threshold',  fmt(modelInfo.decision_threshold, 6)],
                ['Epoch',      modelInfo.epoch ?? '—'],
              ].map(([k, v]) => <MetricRow key={k} label={k} value={v} />)}
            </div>
          )}
          {!ready && (
            <div style={{ background: 'var(--yellow-dim)', border: '1px solid rgba(245,158,11,0.3)', borderRadius: 'var(--radius-sm)', padding: 14, textAlign: 'center' }}>
              <div style={{ color: 'var(--yellow)', fontWeight: 700, marginBottom: 4 }}>⚠ MODEL OFFLINE</div>
              <div style={{ color: 'var(--text-muted)', fontSize: 'var(--fs-sm)' }}>Train with: <code style={{ fontFamily: 'var(--font-mono)' }}>python -m training.train --asvspoof5-dir &lt;PATH&gt;</code></div>
            </div>
          )}
        </div>

        {/* Section 2: Architecture diagram */}
        <div className="card" style={{ marginBottom: 16 }}>
          <SectionTitle>Architecture Diagram</SectionTitle>
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
            <div className="arch-box" style={{ background: 'rgba(37,99,235,0.12)', borderColor: 'rgba(37,99,235,0.4)', color: 'var(--accent)', fontWeight: 700 }}>RAW AUDIO INPUT (16 kHz mono, 4s)</div>
            <div className="arch-arrow">↓</div>
            <div style={{ display: 'flex', gap: 16, alignItems: 'flex-start' }}>
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
                <div className="arch-box" style={{ flexDirection: 'column', textAlign: 'center' }}>
                  <div style={{ fontWeight: 600, color: 'var(--text-primary)' }}>1D CNN BRANCH</div>
                  <div style={{ marginTop: 2 }}>4-layer, residual</div>
                  <div>kernels 15/9/7/5, GELU, BN</div>
                </div>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
                <div className="arch-box" style={{ flexDirection: 'column', textAlign: 'center' }}>
                  <div style={{ fontWeight: 600, color: 'var(--text-primary)' }}>LOG-MEL BRANCH</div>
                  <div style={{ marginTop: 2 }}>4-layer 2D CNN, residual</div>
                  <div>80 mel bins, 512-pt FFT</div>
                </div>
              </div>
            </div>
            <div className="arch-arrow">↓</div>
            <div className="arch-box" style={{ fontWeight: 600 }}>FEATURE FUSION — Concatenate</div>
            <div className="arch-arrow">↓</div>
            <div className="arch-box">BiGRU — 2 layers, hidden=256, bidirectional</div>
            <div className="arch-arrow">↓</div>
            <div className="arch-box">TEMPORAL ATTENTION — Learnable</div>
            <div className="arch-arrow">↓</div>
            <div className="arch-box">CLASSIFIER — 3-layer MLP → 1 logit</div>
            <div className="arch-arrow">↓</div>
            <div className="arch-box">sigmoid(logit) = SPOOF PROBABILITY</div>
            <div className="arch-arrow">↓</div>
            <div className="arch-box" style={{ background: 'rgba(239,68,68,0.1)', borderColor: 'rgba(239,68,68,0.35)' }}>RISK ENGINE → THREAT LEVEL</div>
          </div>
          <div style={{ fontSize: 'var(--fs-xxs)', color: 'var(--text-muted)', marginTop: 12, textAlign: 'center' }}>~3,833,218 parameters · BCEWithLogitsLoss · Trained from random initialization</div>
        </div>

        {/* Section 3: Validation metrics */}
        <div className="card" style={{ marginBottom: 16 }}>
          <SectionTitle icon={<BarChart3 size={13} />}>Validation Metrics (Speaker-Disjoint Internal Validation)</SectionTitle>
          <div style={{ background: 'var(--yellow-dim)', border: '1px solid rgba(245,158,11,0.25)', borderRadius: 'var(--radius-sm)', padding: '8px 12px', marginBottom: 14, fontSize: 'var(--fs-xs)', color: 'var(--yellow)' }}>
            ⚠ These are validation-split results. Not real-world performance guarantees.
          </div>
          {!ready ? (
            <div style={{ color: 'var(--text-muted)', fontSize: 'var(--fs-sm)', textAlign: 'center', padding: '16px 0' }}>Model offline — metrics unavailable</div>
          ) : Object.keys(vm).length === 0 ? (
            <div style={{ color: 'var(--text-muted)', fontSize: 'var(--fs-sm)', textAlign: 'center', padding: '16px 0' }}>No validation metrics available</div>
          ) : (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))', gap: 8 }}>
              {[['ROC-AUC', vm.roc_auc], ['EER', vm.eer], ['F1', vm.f1], ['Recall', vm.recall], ['FPR', vm.fpr], ['FNR', vm.fnr], ['Accuracy', vm.accuracy], ['Precision', vm.precision]].filter(([, v]) => v !== undefined).map(([k, v]) => (
                <div key={k} style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', padding: '10px 12px' }}>
                  <div style={{ fontSize: 'var(--fs-xxs)', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em', marginBottom: 4 }}>{k}</div>
                  <div style={{ fontSize: 18, fontFamily: 'var(--font-mono)', fontWeight: 700, color: 'var(--text-primary)' }}>{fmt(v, 4)}</div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Section 4: Risk engine */}
        <div className="card" style={{ marginBottom: 16 }}>
          <SectionTitle>Risk Classification Policy</SectionTitle>
          <div style={{ fontSize: 'var(--fs-xs)', color: 'var(--text-muted)', marginBottom: 10 }}>Deterministic policy — not a second ML model</div>
          <table className="data-table" style={{ marginBottom: 12 }}>
            <thead><tr><th>Spoof Probability</th><th>Threat Level</th></tr></thead>
            <tbody>
              {[['≥ 0.90', 'CRITICAL'], ['≥ 0.70', 'HIGH'], ['≥ 0.45', 'MEDIUM'], ['< 0.45', 'LOW']].map(([cond, level]) => (
                <tr key={level}>
                  <td style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>{cond}</td>
                  <td><ThreatBadge level={level} /></td>
                </tr>
              ))}
            </tbody>
          </table>
          <div style={{ fontSize: 'var(--fs-sm)', color: 'var(--text-secondary)', background: 'var(--bg-secondary)', borderRadius: 'var(--radius-sm)', padding: '8px 12px', fontFamily: 'var(--font-mono)' }}>
            Risk Score = spoof_probability ^ 0.7 (monotone, emphasises high-end risk)
          </div>
        </div>

        {/* Section 5: Detection scope */}
        <div className="scope-card">
          <div style={{ fontSize: 'var(--fs-sm)', fontWeight: 700, color: 'var(--yellow)', marginBottom: 10 }}>Detection Scope</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6, fontSize: 'var(--fs-xs)' }}>
            {[
              { icon: '✓', color: 'var(--green)', text: 'DETECTS: Direct digital TTS' },
              { icon: '✓', color: 'var(--green)', text: 'DETECTS: Voice conversion (logical-access spoofing)' },
              { icon: '✓', color: 'var(--green)', text: 'DETECTS: ASVspoof5 attack types' },
              { icon: '⚠', color: 'var(--yellow)', text: 'CURRENT ROBUSTNESS TARGET: Acoustic replay through physical speaker + microphone' },
              { icon: '📊', color: 'var(--accent)', text: 'TRAINED ON: ASVspoof5 training set, random initialization' },
              { icon: '⚠', color: 'var(--yellow)', text: 'NOT VALIDATED ON: Live phone networks, codecs outside ASVspoof5 coverage' },
            ].map(({ icon, color, text }) => (
              <div key={text} style={{ display: 'flex', gap: 8, color: 'var(--text-secondary)' }}>
                <span style={{ color, flexShrink: 0 }}>{icon}</span>
                <span>{text}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// DemoModePage
// ─────────────────────────────────────────────────────────────────────────────

function DemoModePage({ stats, modelInfo, onNavigate }) {
  const ready = modelInfo?.status === 'ready'
  const vm    = modelInfo?.val_metrics || {}

  const steps = [
    {
      num: '01', title: 'REAL VOICE',
      desc: 'Test a live microphone input. Expected result: BONA_FIDE.',
      btn: 'Go to Live Monitor →', page: 'Live Monitor',
      extra: null,
    },
    {
      num: '02', title: 'DIGITAL SPOOF',
      desc: 'Upload a verified spoof audio file. Expected result: SPOOF.',
      btn: 'Go to Analyze Voice →', page: 'Analyze Voice',
      extra: (
        <div style={{ fontSize: 'var(--fs-xs)', color: 'var(--text-muted)', marginTop: 8, background: 'var(--bg-secondary)', borderRadius: 'var(--radius-sm)', padding: '8px 10px', lineHeight: 'var(--lh-body)' }}>
          <div style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)', marginBottom: 4 }}>demo_audio/spoof_voice.flac</div>
          Place <code style={{ fontFamily: 'var(--font-mono)' }}>demo_audio/spoof_voice.flac</code> in the VoxShield root, then upload via Analyze Voice.
        </div>
      ),
    },
    {
      num: '03', title: 'INSPECT INCIDENT',
      desc: 'After analyses, inspect the resulting incidents in the log.',
      btn: 'Go to Incidents →', page: 'Incidents',
      extra: <div style={{ fontSize: 'var(--fs-xs)', color: 'var(--text-muted)', marginTop: 6 }}>Current incidents: <strong style={{ color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }}>{stats?.total ?? 0}</strong></div>,
    },
    {
      num: '04', title: 'MODEL DETAILS',
      desc: 'Show the actual validation metrics and architecture.',
      btn: 'Go to Model Performance →', page: 'Model Performance',
      extra: (
        <div style={{ fontSize: 'var(--fs-xs)', color: 'var(--text-muted)', marginTop: 6 }}>
          Status: <strong style={{ color: ready ? 'var(--green)' : 'var(--yellow)' }}>{ready ? 'READY' : 'OFFLINE'}</strong>
          {ready && vm.roc_auc != null && <> · ROC-AUC: <strong style={{ color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }}>{fmt(vm.roc_auc, 4)}</strong></>}
          {ready && vm.eer != null && <> · EER: <strong style={{ color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }}>{fmt(vm.eer, 4)}</strong></>}
        </div>
      ),
    },
  ]

  return (
    <div style={{ padding: '24px 28px' }}>
      <div className="page-header" style={{ margin: '-24px -28px 24px', padding: '24px 28px' }}>
        <h1 style={{ fontSize: 'var(--fs-h1)', fontWeight: 700, color: 'var(--text-primary)', margin: 0 }}>Demo Mode — Guided Presentation Flow</h1>
        <p style={{ fontSize: 'var(--fs-body)', color: 'var(--text-secondary)', marginTop: 4 }}>Walk judges through the product in 4 steps. All inference is real.</p>
      </div>

      <div style={{ background: 'var(--yellow-dim)', border: '1px solid rgba(245,158,11,0.3)', borderRadius: 'var(--radius)', padding: '10px 16px', marginBottom: 24, fontSize: 'var(--fs-sm)', color: 'var(--yellow)', display: 'flex', gap: 8, alignItems: 'center' }}>
        <AlertTriangle size={16} />
        Demo mode uses real inference only. Results depend on the actual model and audio. No results are simulated.
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: 16, maxWidth: 860 }}>
        {steps.map(({ num, title, desc, btn, page, extra }) => (
          <div key={num} className="demo-step">
            <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 12 }}>
              <div className="demo-step-number">{num}</div>
              <div style={{ fontSize: 'var(--fs-body)', fontWeight: 700, color: 'var(--text-primary)' }}>{title}</div>
            </div>
            <p style={{ fontSize: 'var(--fs-sm)', color: 'var(--text-secondary)', lineHeight: 'var(--lh-body)', marginBottom: 12 }}>{desc}</p>
            {extra}
            <button className="btn btn-primary" style={{ width: '100%', marginTop: 12, padding: '9px' }} onClick={() => onNavigate(page)}>{btn}</button>
          </div>
        ))}
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// SettingsPage
// ─────────────────────────────────────────────────────────────────────────────

function SettingsPage({ health, modelInfo }) {
  const ready = modelInfo?.status === 'ready'
  return (
    <div style={{ padding: '24px 28px' }}>
      <div className="page-header" style={{ margin: '-24px -28px 24px', padding: '24px 28px' }}>
        <h1 style={{ fontSize: 'var(--fs-h1)', fontWeight: 700, color: 'var(--text-primary)', margin: 0 }}>System & Privacy</h1>
        <p style={{ fontSize: 'var(--fs-body)', color: 'var(--text-secondary)', marginTop: 4 }}>Backend connection, detection configuration, privacy and system information.</p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, maxWidth: 760 }}>
        {/* Connection */}
        <div className="card">
          <SectionTitle icon={<Shield size={13} />}>Connection</SectionTitle>
          <MetricRow label="Backend URL"   value="http://localhost:8000" />
          <MetricRow label="API Status"    value={health ? 'Connected' : 'Offline'} mono={false} />
          <MetricRow label="Model Status"  value={ready ? 'Ready' : 'Offline'} mono={false} />
          <MetricRow label="Poll (data)"   value="5 000 ms" />
          <MetricRow label="Poll (incidents)" value="3 000 ms" />
        </div>

        {/* Live detection params */}
        <div className="card">
          <SectionTitle icon={<Mic size={13} />}>Live Detection Parameters</SectionTitle>
          <MetricRow label="Window Size"       value="4 seconds" />
          <MetricRow label="Hop Interval"      value="2000 ms" />
          <MetricRow label="Sample Rate"       value="16 000 Hz" />
          <MetricRow label="EMA Alpha"         value="0.35" />
          <MetricRow label="EMA Window"        value="5 samples" />
        </div>

        {/* Privacy */}
        <div className="card">
          <SectionTitle icon={<Shield size={13} />}>Privacy & Data</SectionTitle>
          <MetricRow label="External API calls" value="None" mono={false} />
          <MetricRow label="Audio retention"    value="In-memory only" mono={false} />
          <MetricRow label="Incident log"        value="Local SQLite" mono={false} />
          <MetricRow label="Audio data"          value="Never leaves this machine" mono={false} />
        </div>

        {/* About */}
        <div className="card">
          <SectionTitle icon={<Cpu size={13} />}>About</SectionTitle>
          <MetricRow label="Version"        value="VoxShield Independent v1.0" mono={false} />
          <MetricRow label="VoxShieldNet"   value="3,833,218 parameters" />
          <MetricRow label="Training"       value="ASVspoof5" mono={false} />
          <MetricRow label="Pretrained"     value="None" mono={false} />
          <MetricRow label="Build"          value="SIH Demo" mono={false} />
        </div>
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Main App
// ─────────────────────────────────────────────────────────────────────────────

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

  const refreshData = useCallback(async () => {
    try {
      const [h, mi] = await Promise.all([api.health(), api.modelInfo()])
      setHealth(h); setModelInfo(mi); setApiError(false)
    } catch { setApiError(true) }
  }, [])

  const refreshIncidents = useCallback(async () => {
    try {
      const [tl, inc, st] = await Promise.all([
        api.timeline(20), api.incidents({ limit: 100 }), api.incidentStats(),
      ])
      setTimeline(tl.timeline || [])
      setIncidents(inc.incidents || [])
      setStats(st)
    } catch { /* silent */ }
  }, [])

  useEffect(() => {
    refreshData(); refreshIncidents()
    const t1 = setInterval(refreshData, 5000)
    const t2 = setInterval(refreshIncidents, 3000)
    return () => { clearInterval(t1); clearInterval(t2) }
  }, [refreshData, refreshIncidents])

  function handleResult(r) {
    setResult(r)
    setTimeout(refreshIncidents, 500)
  }

  function renderPage() {
    switch (active) {
      case 'Dashboard':
        return <DashboardPage health={health} modelInfo={modelInfo} result={result} timeline={timeline} incidents={incidents} stats={stats} onResult={handleResult} onReload={refreshData} onNavigate={setActive} />
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
      case 'Demo Mode':
        return <DemoModePage stats={stats} modelInfo={modelInfo} onNavigate={setActive} />
      case 'System & Privacy':
        return <SettingsPage health={health} modelInfo={modelInfo} />
      case 'Settings':
        return <SettingsPage health={health} modelInfo={modelInfo} />
      default:
        return <DashboardPage health={health} modelInfo={modelInfo} result={result} timeline={timeline} incidents={incidents} stats={stats} onResult={handleResult} onReload={refreshData} onNavigate={setActive} />
    }
  }

  return (
    <div style={{ display: 'flex', height: '100vh', overflow: 'hidden', background: 'var(--bg-page)' }}>
      <Sidebar active={active} setActive={setActive} />
      <main style={{ marginLeft: 240, width: 'calc(100% - 240px)', display: 'flex', flexDirection: 'column', height: '100vh', overflow: 'hidden' }}>
        <Topbar modelReady={modelReady} />
        {apiError && (
          <div style={{ background: 'var(--danger-bg)', borderBottom: '1px solid var(--danger-border)', padding: '9px 24px', color: 'var(--danger)', fontSize: 'var(--fs-sm)', textAlign: 'center', flexShrink: 0 }}>
            ⚠ Cannot connect to VoxShield API (localhost:8000) — Start the backend:
            <code style={{ marginLeft: 10, fontFamily: 'var(--font-mono)', fontSize: 'var(--fs-xs)' }}>uvicorn backend.app:app --port 8000</code>
          </div>
        )}
        {active === 'Dashboard' ? (
          <div style={{ flex: 1, minHeight: 0, overflow: 'hidden' }}>{renderPage()}</div>
        ) : (
          <div style={{ flex: 1, overflowY: 'auto' }}>
            {renderPage()}
            <footer style={{
              borderTop: '1px solid var(--border)',
              padding: '9px 24px',
              display: 'flex', justifyContent: 'space-between', alignItems: 'center',
              fontSize: 'var(--fs-nano)', color: 'var(--text-muted)',
              flexWrap: 'wrap', gap: 4,
              background: 'var(--bg-card)',
            }}>
              <span>VoxShield Independent v1.0 · Voice Threat Intelligence · Local Inference Only</span>
              <span>VoxShieldNet trained from random initialization · No pretrained components · No external APIs</span>
              <span>© 2026 · SIH Demo Build</span>
            </footer>
          </div>
        )}
      </main>
    </div>
  )
}
