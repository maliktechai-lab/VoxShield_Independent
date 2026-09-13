/**
 * VoxShield API client
 * All calls go to local backend (no external AI APIs).
 */

const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

async function apiFetch(path, options = {}) {
  const res = await fetch(`${BASE_URL}${path}`, options)
  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = await res.json()
      detail = body.detail || body.error || detail
    } catch (_) {}
    throw new Error(`API ${path}: ${res.status} — ${detail}`)
  }
  return res.json()
}

export const api = {
  health: ()         => apiFetch('/health'),
  modelInfo: ()      => apiFetch('/model-info'),
  reloadModel: ()    => apiFetch('/reload-model', { method: 'POST' }),
  incidents: (p = {}) => {
    const q = new URLSearchParams()
    if (p.limit)          q.set('limit',          p.limit)
    if (p.offset)         q.set('offset',         p.offset)
    if (p.classification) q.set('classification', p.classification)
    if (p.threat_level)   q.set('threat_level',   p.threat_level)
    return apiFetch(`/incidents?${q}`)
  },
  incidentStats: ()  => apiFetch('/incidents/stats'),
  timeline: (n = 20) => apiFetch(`/incidents/timeline?n=${n}`),

  /** Upload prediction — used by the manual upload panel. */
  predict: (file) => {
    const form = new FormData()
    form.append('file', file)
    return apiFetch('/predict', { method: 'POST', body: form })
  },

  /**
   * Live mic prediction — same backend endpoint but passes session_id
   * so live-session incidents are distinguishable in the incident log
   * (filename prefix "live_mic_" also serves as a secondary marker).
   */
  predictLive: (file, sessionId) => {
    const form = new FormData()
    form.append('file', file)
    if (sessionId) form.append('session_id', sessionId)
    return apiFetch('/predict', { method: 'POST', body: form })
  },
}

export default api
