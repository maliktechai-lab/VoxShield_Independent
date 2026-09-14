/**
 * VoxShield Live Microphone Detection Overlay
 * =============================================
 *
 * Design:
 *   - Rolling 8-second audio buffer at native browser sample rate
 *   - Every HOP_MS (2s), extract the most recent 4 seconds
 *   - Linear resample to 16 kHz
 *   - Encode as PCM-16 WAV
 *   - POST to /predict with session_id tag "live_<uuid>"
 *   - Display raw model spoof_probability + smoothed 5-window EMA verdict
 *
 * Smoothing (Phase 4):
 *   A lightweight exponential moving average (alpha=0.35) over the last
 *   N_SMOOTH windows is applied ONLY to the displayed verdict decision.
 *   The raw per-window spoof_probability is always displayed unchanged.
 *   The model threshold (0.6729) is never altered.
 *   Smoothed verdict = smoothedProb >= threshold → SPOOF
 *
 * Incident logging (Phase 5):
 *   All /predict calls from live mode include session_id="live_<uuid>".
 *   This allows filtering live vs uploaded incidents in the incident log.
 *   The backend logs every call — we intentionally do not suppress logging
 *   but the filename prefix "live_mic_" distinguishes live vs upload.
 *
 * Error handling:
 *   - Microphone denied / not available
 *   - Unsupported browser
 *   - Backend unavailable
 *   - Request timeout (5 seconds hard limit)
 *   - Malformed backend response
 *   - Network errors
 *   - Audio context failure
 *
 * ScriptProcessor note:
 *   createScriptProcessor is deprecated but still works in all major browsers
 *   (Chrome, Edge, Firefox) as of 2026. No AudioWorklet polyfill needed for demo.
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from './api.js'

// ── Constants ──────────────────────────────────────────────────────────────────

const WINDOW_SECONDS   = 4          // model input window length
const HOP_MS           = 2000       // ms between successive inference calls
const BUFFER_SECONDS   = 8          // rolling audio buffer size
const TARGET_SAMPLE_RATE = 16000    // model sample rate
const SCRIPT_PROC_SIZE = 4096       // ScriptProcessor buffer size
const INFERENCE_TIMEOUT_MS = 5000   // max ms to wait for /predict response
const N_SMOOTH         = 5          // EMA window for smoothed verdict
const EMA_ALPHA        = 0.35       // smoothing factor (higher = more reactive)
const LEVEL_UPDATE_MS  = 100        // mic level meter update interval

// ── Helpers ────────────────────────────────────────────────────────────────────

function probabilityPct(value) {
  if (value === null || value === undefined) return 'N/A'
  if (value <= 0) return '<0.01%'
  if (value >= 1) return '>99.99%'
  return `${(value * 100).toFixed(2)}%`
}

function clamp01(value) {
  return Math.max(0, Math.min(1, Number(value) || 0))
}

/**
 * Extract the most recent `sampleCount` samples from the rolling chunk list.
 * Iterates backward through chunks to avoid copying the full buffer.
 */
function takeRecentSamples(chunks, totalSamples, sampleCount) {
  const count = Math.min(sampleCount, totalSamples)
  const out = new Float32Array(count)
  let writeOffset = count
  let remaining = count

  for (let i = chunks.length - 1; i >= 0 && remaining > 0; i -= 1) {
    const chunk = chunks[i]
    const take = Math.min(chunk.length, remaining)
    writeOffset -= take
    out.set(chunk.subarray(chunk.length - take), writeOffset)
    remaining -= take
  }

  return out
}

/**
 * Linear interpolation resampler.
 * Good enough for 44100→16000 and 48000→16000 for demo purposes.
 * Does not apply an anti-aliasing low-pass filter before decimation,
 * which may introduce mild aliasing, but does not affect correctness
 * of the WAV format sent to the backend.
 */
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

/**
 * Encode float32 PCM samples as a valid RIFF/PCM-16 WAV blob.
 * The backend AudioPreprocessor accepts this via soundfile.read().
 */
function encodeWavPcm16(samples, sampleRate) {
  const dataSize = samples.length * 2
  const buffer = new ArrayBuffer(44 + dataSize)
  const view = new DataView(buffer)

  const ws = (offset, s) => {
    for (let i = 0; i < s.length; i++) view.setUint8(offset + i, s.charCodeAt(i))
  }

  ws(0,  'RIFF')
  view.setUint32(4,  36 + dataSize, true)
  ws(8,  'WAVE')
  ws(12, 'fmt ')
  view.setUint32(16, 16, true)      // PCM
  view.setUint16(20, 1,  true)      // AudioFormat = PCM
  view.setUint16(22, 1,  true)      // NumChannels = mono
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, sampleRate * 2, true)  // ByteRate
  view.setUint16(32, 2,  true)      // BlockAlign
  view.setUint16(34, 16, true)      // BitsPerSample
  ws(36, 'data')
  view.setUint32(40, dataSize, true)

  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]))
    view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true)
  }

  return new Blob([buffer], { type: 'audio/wav' })
}

function makeLiveFilename() {
  const stamp = new Date().toISOString().replace(/[-:.TZ]/g, '').slice(0, 14)
  return `live_mic_${stamp}.wav`
}

/** Generate a stable session ID for this live detection session. */
function makeSessionId() {
  return 'live_' + Math.random().toString(36).slice(2, 10)
}

/** EMA update: newEma = alpha * newValue + (1 - alpha) * prevEma */
function emaUpdate(prev, value, alpha) {
  if (prev === null) return value
  return alpha * value + (1 - alpha) * prev
}

// ── Sub-components ─────────────────────────────────────────────────────────────

function StatusPill({ active, analyzing }) {
  const label  = analyzing ? 'ANALYZING' : active ? 'MIC LIVE' : 'MIC OFF'
  const color  = analyzing ? 'var(--accent)' : active ? 'var(--red)' : 'var(--text-muted)'
  const bg     = analyzing ? 'rgba(99,102,241,0.15)' : active ? 'var(--red-dim)' : 'var(--bg-secondary)'
  const border = analyzing ? 'rgba(99,102,241,0.3)' : active ? 'rgba(239,68,68,0.3)' : 'var(--border)'
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: 5,
      fontSize: 11, fontWeight: 700, letterSpacing: '0.08em',
      padding: '4px 10px', borderRadius: 4,
      color, background: bg, border: `1px solid ${border}`,
    }}>
      <span className={active ? 'pulse' : ''} style={{
        width: 7, height: 7, borderRadius: '50%', background: color, flexShrink: 0,
      }} />
      {label}
    </span>
  )
}

function ProbBar({ value }) {
  const pct = clamp01(value ?? 0) * 100
  const color = pct >= 67 ? 'var(--red)' : pct >= 45 ? '#f59e0b' : 'var(--green)'
  return (
    <div style={{
      height: 5, background: 'var(--bg-secondary)',
      borderRadius: 3, overflow: 'hidden', marginTop: 4,
    }}>
      <div style={{ width: `${pct}%`, height: '100%', background: color, transition: 'width 0.3s ease' }} />
    </div>
  )
}

// ── Main Component ─────────────────────────────────────────────────────────────

export default function LiveDetectionOverlay() {
  const [active,          setActive]          = useState(false)
  const [status,          setStatus]          = useState('idle')
  const [error,           setError]           = useState(null)
  const [level,           setLevel]           = useState(0)
  const [windowProgress,  setWindowProgress]  = useState(0)
  const [liveResult,      setLiveResult]      = useState(null)
  const [lastLatency,     setLastLatency]      = useState(null)
  const [smoothedProb,    setSmoothedProb]     = useState(null)
  const [windowCount,     setWindowCount]      = useState(0)
  const [recentHistory,   setRecentHistory]    = useState([])   // last 5 probs

  // Audio refs (not state — mutated in audio thread without triggering renders)
  const streamRef        = useRef(null)
  const audioContextRef  = useRef(null)
  const sourceRef        = useRef(null)
  const processorRef     = useRef(null)
  const muteGainRef      = useRef(null)
  const timerRef         = useRef(null)
  const chunksRef        = useRef([])
  const totalSamplesRef  = useRef(0)
  const analyzingRef     = useRef(false)
  const startedAtRef     = useRef(0)
  const nextAnalysisAtRef = useRef(0)
  const lastLevelUpdateRef = useRef(0)
  const smoothedProbRef  = useRef(null)   // EMA state, kept in ref for the timer
  const sessionIdRef     = useRef(null)   // stable session ID per live session

  // ── Cleanup ──────────────────────────────────────────────────────────────────

  const cleanup = useCallback(async () => {
    if (timerRef.current) {
      clearInterval(timerRef.current)
      timerRef.current = null
    }

    // Disconnect audio graph before closing context
    try { processorRef.current?.disconnect() } catch (_) {}
    try { sourceRef.current?.disconnect() }    catch (_) {}
    try { muteGainRef.current?.disconnect() }  catch (_) {}

    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop())
    }

    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      try { await audioContextRef.current.close() } catch (_) {}
    }

    // Reset all refs
    streamRef.current        = null
    audioContextRef.current  = null
    sourceRef.current        = null
    processorRef.current     = null
    muteGainRef.current      = null
    chunksRef.current        = []
    totalSamplesRef.current  = 0
    analyzingRef.current     = false
    startedAtRef.current     = 0
    nextAnalysisAtRef.current = 0
    smoothedProbRef.current  = null
    sessionIdRef.current     = null

    // Reset state
    setActive(false)
    setStatus('idle')
    setLevel(0)
    setWindowProgress(0)
    setSmoothedProb(null)
  }, [])

  // ── Analysis window ──────────────────────────────────────────────────────────

  const analyzeWindow = useCallback(async () => {
    // Guard: skip if already analyzing or audio context gone
    if (!audioContextRef.current || analyzingRef.current) return

    const sourceRate = audioContextRef.current.sampleRate
    const requiredSamples = Math.floor(sourceRate * WINDOW_SECONDS)
    if (totalSamplesRef.current < requiredSamples) return

    analyzingRef.current = true
    setStatus('analyzing')

    try {
      // Extract most recent WINDOW_SECONDS of samples
      const recent = takeRecentSamples(
        chunksRef.current,
        totalSamplesRef.current,
        requiredSamples,
      )

      // Resample to 16 kHz
      let fixedWindow = resampleLinear(recent, sourceRate, TARGET_SAMPLE_RATE)

      // Enforce exact 64000 sample count (pad or trim)
      const expected = TARGET_SAMPLE_RATE * WINDOW_SECONDS
      if (fixedWindow.length > expected) {
        fixedWindow = fixedWindow.slice(0, expected)
      } else if (fixedWindow.length < expected) {
        const padded = new Float32Array(expected)
        padded.set(fixedWindow)
        fixedWindow = padded
      }

      // Encode as PCM-16 WAV
      const wav = encodeWavPcm16(fixedWindow, TARGET_SAMPLE_RATE)
      const wavFile = new File([wav], makeLiveFilename(), { type: 'audio/wav' })

      // POST /predict with timeout and live session_id
      const controller = new AbortController()
      const timeoutId = setTimeout(() => controller.abort(), INFERENCE_TIMEOUT_MS)

      let result
      try {
        result = await api.predictLive(wavFile, sessionIdRef.current)
      } finally {
        clearTimeout(timeoutId)
      }

      // Validate response has expected fields
      if (!result || typeof result.spoof_probability !== 'number') {
        throw new Error('Malformed response from /predict')
      }

      const rawProb = clamp01(result.spoof_probability)

      // Update EMA smoothed probability
      const newSmoothed = emaUpdate(smoothedProbRef.current, rawProb, EMA_ALPHA)
      smoothedProbRef.current = newSmoothed

      // Update window count and recent history
      setWindowCount(c => c + 1)
      setRecentHistory(prev => {
        const next = [...prev, rawProb]
        return next.slice(-N_SMOOTH)
      })

      // Apply smoothed prob for verdict display
      setSmoothedProb(newSmoothed)
      setLiveResult(result)
      setLastLatency(result.latency_ms ?? null)
      setError(null)
    } catch (e) {
      if (e.name === 'AbortError') {
        setError('Inference timeout (>5s) — backend may be slow')
      } else {
        setError(e.message || 'Unknown inference error')
      }
      // Don't set status to permanent error — keep listening
    } finally {
      analyzingRef.current = false
      // Only restore 'listening' if still connected
      if (streamRef.current && audioContextRef.current) {
        setStatus('listening')
      }
    }
  }, [])

  // ── Start microphone ─────────────────────────────────────────────────────────

  const start = useCallback(async () => {
    if (active) return

    setError(null)
    setLiveResult(null)
    setLastLatency(null)
    setSmoothedProb(null)
    setWindowCount(0)
    setRecentHistory([])
    setStatus('requesting')

    // Security check — mic requires secure context or localhost
    if (!window.isSecureContext &&
        window.location.hostname !== 'localhost' &&
        window.location.hostname !== '127.0.0.1') {
      setError('Microphone requires HTTPS or localhost.')
      setStatus('error')
      return
    }

    if (!navigator.mediaDevices?.getUserMedia) {
      setError('getUserMedia not supported. Use Chrome/Edge/Firefox.')
      setStatus('error')
      return
    }

    try {
      // Request mono microphone — disable browser enhancements so we
      // get the raw signal the model was trained to expect
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount:      1,
          echoCancellation:  false,
          noiseSuppression:  false,
          autoGainControl:   false,
          sampleRate:        { ideal: 48000 },  // hint, not guaranteed
        },
      })

      const AudioContextClass = window.AudioContext || window.webkitAudioContext
      if (!AudioContextClass) throw new Error('Web Audio API unavailable.')

      const context = new AudioContextClass()
      // Safari/iOS may start in 'suspended' state
      if (context.state === 'suspended') await context.resume()
      if (context.state !== 'running') throw new Error('AudioContext failed to start.')

      const source    = context.createMediaStreamSource(stream)
      // ScriptProcessor: deprecated but universally supported in 2026
      // bufferSize=4096 → ~85ms latency at 48kHz (negligible vs 4s window)
      const processor = context.createScriptProcessor(SCRIPT_PROC_SIZE, 1, 1)
      // Mute gain prevents feedback loop through speakers
      const muteGain  = context.createGain()
      muteGain.gain.value = 0

      processor.onaudioprocess = event => {
        // Capture input samples
        const input = event.inputBuffer.getChannelData(0)
        const copy = new Float32Array(input)
        chunksRef.current.push(copy)
        totalSamplesRef.current += copy.length

        // Trim ring buffer to BUFFER_SECONDS
        const maxSamples = Math.floor(context.sampleRate * BUFFER_SECONDS)
        while (totalSamplesRef.current > maxSamples && chunksRef.current.length > 1) {
          const removed = chunksRef.current.shift()
          totalSamplesRef.current -= removed.length
        }

        // Update RMS level meter (throttled)
        const now = performance.now()
        if (now - lastLevelUpdateRef.current > LEVEL_UPDATE_MS) {
          lastLevelUpdateRef.current = now
          let ss = 0
          for (let i = 0; i < input.length; i++) ss += input[i] * input[i]
          setLevel(clamp01(Math.sqrt(ss / Math.max(1, input.length)) * 8))
        }
      }

      // Connect graph: source → processor → muteGain → destination
      source.connect(processor)
      processor.connect(muteGain)
      muteGain.connect(context.destination)

      // Store refs
      streamRef.current         = stream
      audioContextRef.current   = context
      sourceRef.current         = source
      processorRef.current      = processor
      muteGainRef.current       = muteGain
      startedAtRef.current      = Date.now()
      nextAnalysisAtRef.current = WINDOW_SECONDS * 1000
      sessionIdRef.current      = makeSessionId()

      setActive(true)
      setStatus('listening')

      // Poll timer: check every 250ms whether it's time to analyze
      timerRef.current = window.setInterval(() => {
        const elapsed = Date.now() - startedAtRef.current
        setWindowProgress(Math.min(1, elapsed / (WINDOW_SECONDS * 1000)))

        if (elapsed >= nextAnalysisAtRef.current) {
          // Schedule next BEFORE calling analyzeWindow so that a slow
          // inference does not cause double-scheduling
          nextAnalysisAtRef.current += HOP_MS
          void analyzeWindow()
        }
      }, 250)

    } catch (e) {
      const msg = e.name === 'NotAllowedError'
        ? 'Microphone permission denied. Enable mic access and try again.'
        : e.name === 'NotFoundError'
        ? 'No microphone found. Connect a microphone and retry.'
        : e.message || 'Failed to start microphone.'
      setError(msg)
      setStatus('error')
      await cleanup()
    }
  }, [active, analyzeWindow, cleanup])

  // Cleanup on unmount
  useEffect(() => () => { void cleanup() }, [cleanup])

  // ── Derived display values ───────────────────────────────────────────────────

  // Raw latest probability (always shown as-is from the model)
  const rawProb = liveResult ? clamp01(liveResult.spoof_probability ?? 0) : null

  // Smoothed verdict uses EMA probability vs the model's own threshold
  // The threshold is read from the backend response (decision_threshold field)
  // and never hardcoded here.
  const modelThreshold = liveResult?.decision_threshold ?? 0.6729
  const smoothedIsSpoof = smoothedProb !== null
    ? smoothedProb >= modelThreshold
    : (rawProb !== null ? rawProb >= modelThreshold : null)

  const verdictText  = smoothedIsSpoof === null ? null : smoothedIsSpoof ? 'SPOOF DETECTED' : 'BONA FIDE'
  const verdictColor = smoothedIsSpoof === null ? 'var(--text-muted)' : smoothedIsSpoof ? 'var(--red)' : 'var(--green)'
  const verdictBg    = smoothedIsSpoof === null ? 'var(--bg-secondary)' : smoothedIsSpoof ? 'var(--red-dim)' : 'var(--green-dim)'

  // Show result panel if we have a valid result (not just status 'ok' — backend
  // successful responses don't have a top-level status field)
  const hasResult = liveResult !== null &&
    liveResult.classification !== undefined &&
    liveResult.classification !== 'ERROR'

  const analyzing = status === 'analyzing'

  return (
    <div style={{
      position:     'fixed',
      right:        18,
      bottom:       18,
      width:        360,
      zIndex:       1000,
      background:   'var(--bg-card)',
      border:       `1px solid ${active ? 'rgba(239,68,68,0.45)' : 'var(--border)'}`,
      borderRadius: 'var(--radius)',
      boxShadow:    active
        ? '0 0 28px rgba(239,68,68,0.14)'
        : '0 12px 28px rgba(0,0,0,0.3)',
      overflow: 'hidden',
      fontFamily: 'var(--font-sans, system-ui)',
    }}>

      {/* Header */}
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '11px 14px', borderBottom: '1px solid var(--border)',
      }}>
        <div>
          <div style={{
            fontSize: 12, fontWeight: 700, letterSpacing: '0.08em',
            color: 'var(--text-primary)', textTransform: 'uppercase',
          }}>
            Live Microphone Detection
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 2, lineHeight: 1.4 }}>
            Rolling 4s windows · VoxShieldNet local inference
          </div>
        </div>
        <StatusPill active={active} analyzing={analyzing} />
      </div>

      {/* Body */}
      <div style={{ padding: 12 }}>

        {/* Meters row */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8, marginBottom: 10 }}>
          {/* Audio level */}
          <div style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', padding: '8px 10px' }}>
            <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 5, letterSpacing: '0.06em', textTransform: 'uppercase' }}>Audio Level</div>
            <div style={{ height: 5, background: 'var(--bg-card)', borderRadius: 3, overflow: 'hidden' }}>
              <div style={{ width: `${level * 100}%`, height: '100%', background: active ? 'var(--accent)' : 'var(--border)', transition: 'width 0.1s' }} />
            </div>
          </div>

          {/* Window progress */}
          <div style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', padding: '8px 10px' }}>
            <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 3, letterSpacing: '0.06em', textTransform: 'uppercase' }}>Window</div>
            <div style={{ fontSize: 13, fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)', lineHeight: 1.3 }}>
              {active ? `${Math.round(windowProgress * WINDOW_SECONDS)}/${WINDOW_SECONDS}s` : '—'}
            </div>
          </div>

          {/* Window count */}
          <div style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', padding: '8px 10px' }}>
            <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 3, letterSpacing: '0.06em', textTransform: 'uppercase' }}>Windows</div>
            <div style={{ fontSize: 13, fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)', lineHeight: 1.3 }}>
              {windowCount}
            </div>
          </div>
        </div>

        {/* Start/Stop button */}
        <button
          className="btn btn-primary"
          style={{ width: '100%', padding: '11px', fontSize: 13 }}
          onClick={() => { if (active) void cleanup(); else void start() }}
          disabled={status === 'requesting'}
        >
          {status === 'requesting'
            ? 'Requesting microphone…'
            : active
            ? '■ Stop Live Detection'
            : '🎙 Start Live Detection'}
        </button>

        {/* Status / error message */}
        <div style={{
          marginTop: 8, fontSize: 12, minHeight: 16,
          color: status === 'error' ? 'var(--red)' : 'var(--text-muted)',
          lineHeight: 1.4,
        }}>
          {error
            ? `⚠ ${error}`
            : analyzing
            ? '⟳ Analyzing latest 4-second window…'
            : active && windowCount === 0
            ? `🎙 Warming up — first verdict in ${Math.round((1 - windowProgress) * WINDOW_SECONDS)}s…`
            : active
            ? '🎙 Listening — verdict updates every ~2s'
            : 'Mic off. No audio captured until you start detection.'}
        </div>

        {/* Live result panel */}
        {hasResult && (
          <div style={{
            marginTop: 10,
            background: verdictBg,
            border: `1px solid ${verdictColor}40`,
            borderRadius: 'var(--radius-sm)',
            padding: '12px 13px',
          }}>
            {/* Verdict + raw prob — strong hierarchy */}
            <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 8 }}>
              <div>
                <div style={{
                  fontSize: 10, color: 'var(--text-muted)',
                  letterSpacing: '0.07em', textTransform: 'uppercase',
                  marginBottom: 3,
                }}>
                  Smoothed Verdict
                </div>
                {/* Verdict — 22px, 800 — the most prominent element */}
                <div style={{
                  fontSize: 22, fontWeight: 800,
                  color: verdictColor, lineHeight: 1.2,
                  letterSpacing: '0.02em',
                }}>
                  {verdictText}
                </div>
              </div>
              <div style={{ textAlign: 'right' }}>
                <div style={{
                  fontSize: 10, color: 'var(--text-muted)',
                  letterSpacing: '0.05em', textTransform: 'uppercase',
                  marginBottom: 3,
                }}>
                  Raw Spoof Prob
                </div>
                {/* Probability — 16px mono, secondary */}
                <div style={{
                  fontSize: 16, fontFamily: 'var(--font-mono)',
                  color: verdictColor, fontWeight: 600, lineHeight: 1.2,
                }}>
                  {probabilityPct(rawProb)}
                </div>
              </div>
            </div>

            {/* Prob bar (raw) */}
            <ProbBar value={rawProb} />

            {/* Smoothed prob + threshold */}
            <div style={{
              display: 'flex', justifyContent: 'space-between',
              marginTop: 8, fontSize: 11, color: 'var(--text-secondary)',
              lineHeight: 1.4,
            }}>
              <span>
                Smoothed: {probabilityPct(smoothedProb)} (EMA α={EMA_ALPHA})
              </span>
              <span>
                Threshold: {modelThreshold.toFixed(4)}
              </span>
            </div>

            {/* Threat level + latency */}
            <div style={{
              display: 'flex', justifyContent: 'space-between',
              marginTop: 6, fontSize: 12, color: 'var(--text-secondary)',
              gap: 8, alignItems: 'center',
            }}>
              <span style={{ fontWeight: 600, letterSpacing: '0.04em' }} title="From model risk engine">
                {liveResult.threat_level ?? '—'}
              </span>
              <span style={{ fontFamily: 'var(--font-mono)' }} title="Backend inference latency">
                {lastLatency !== null ? `${Number(lastLatency).toFixed(0)}ms` : '—'}
              </span>
            </div>

            {/* Recent window history sparkline */}
            {recentHistory.length > 0 && (
              <div style={{ marginTop: 10 }}>
                <div style={{
                  fontSize: 10, color: 'var(--text-muted)',
                  marginBottom: 5, letterSpacing: '0.05em', textTransform: 'uppercase',
                }}>
                  Recent Windows (last {recentHistory.length})
                </div>
                <div style={{ display: 'flex', gap: 3, alignItems: 'flex-end', height: 20 }}>
                  {recentHistory.map((p, i) => {
                    const h = Math.max(3, Math.round(clamp01(p) * 20))
                    const c = p >= modelThreshold ? 'var(--red)' : 'var(--green)'
                    return (
                      <div key={i} style={{
                        flex: 1, height: h, background: c,
                        borderRadius: 2, opacity: 0.7 + 0.3 * (i / recentHistory.length),
                        title: probabilityPct(p),
                      }} />
                    )
                  })}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Detection scope notice — always visible while active */}
        {active && (
          <div style={{
            marginTop: 8,
            padding: '8px 10px',
            background: 'var(--bg-secondary)',
            border: '1px solid var(--border)',
            borderRadius: 'var(--radius-sm)',
            fontSize: 11,
            color: 'var(--text-muted)',
            lineHeight: 1.5,
          }}>
            <div style={{
              fontWeight: 700, marginBottom: 4,
              color: 'var(--text-secondary)',
              fontSize: 11,
              textTransform: 'uppercase',
              letterSpacing: '0.06em',
            }}>
              Detection Scope
            </div>
            <div>Detects: direct digital TTS / voice-conversion files.</div>
            <div style={{ marginTop: 3, color: '#f59e0b' }}>
              Limitation: acoustic replay (TTS played via speaker then
              re-recorded by mic) is not reliably detected — the model
              was trained on ASVspoof5 logical-access attacks only (no
              physical channel).
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
