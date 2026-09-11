"""
VoxShield Risk Engine
======================

The risk engine is a deterministic policy layer, NOT a second ML model.

Pipeline:
    Model probability (sigmoid output)
    → Risk scoring policy
    → Threat level
    → Recommended action

This is a transparent, auditable mapping from model probability to
operational risk classification. The UI should make this distinction clear.

Threat levels:
    LOW      — high confidence bonafide
    MEDIUM   — borderline or low-confidence result
    HIGH     — high spoof probability
    CRITICAL — very high spoof probability (near certain synthetic)

The thresholds in this policy are calibrated to be conservative for security
use cases — when in doubt, escalate risk rather than suppress it.
"""

from __future__ import annotations

from typing import Dict


class RiskEngine:
    """
    Deterministic risk scoring from model spoof probability.

    This is a policy engine. It does NOT contain another ML model.
    It transforms a single float (spoof_probability) into structured
    risk intelligence suitable for a Security Operations Center (SOC).
    """

    # ── Risk thresholds (tunable per deployment) ──────────────────────────────
    # These are probability thresholds, NOT detection thresholds.
    # Detection threshold (EER-based) is applied BEFORE this engine.

    CRITICAL_THRESHOLD = 0.90   # > 90 % spoof probability
    HIGH_THRESHOLD     = 0.70   # > 70 %
    MEDIUM_THRESHOLD   = 0.45   # > 45 %
    # Below 0.45 → LOW risk

    # ── Risk score ranges ─────────────────────────────────────────────────────
    # Risk score [0.0, 1.0] — continuous, monotone with spoof probability.

    @staticmethod
    def score(spoof_probability: float, decision_threshold: float = 0.5) -> Dict:
        """
        Compute risk from spoof probability.

        Args:
            spoof_probability:  Model sigmoid output ∈ [0, 1].
                                1 = certain spoof, 0 = certain bonafide.
            decision_threshold: The EER-calibrated decision threshold from
                                the trained checkpoint. Used to compute
                                margin from the decision boundary.

        Returns:
            Dict with: risk_score, threat_level, recommended_action, breakdown.
        """
        p = float(spoof_probability)
        p = max(0.0, min(1.0, p))  # clamp

        # ── Risk score ────────────────────────────────────────────────────────
        # Non-linear: emphasise the high-probability end.
        # risk_score = p^0.7  (concave — inflates risk for moderate scores)
        risk_score = round(p ** 0.7, 4)

        # ── Threat level ──────────────────────────────────────────────────────
        if p >= RiskEngine.CRITICAL_THRESHOLD:
            threat_level = "CRITICAL"
        elif p >= RiskEngine.HIGH_THRESHOLD:
            threat_level = "HIGH"
        elif p >= RiskEngine.MEDIUM_THRESHOLD:
            threat_level = "MEDIUM"
        else:
            threat_level = "LOW"

        # ── Recommended action ────────────────────────────────────────────────
        action = RiskEngine._recommended_action(threat_level, p, decision_threshold)

        # ── Breakdown (explainability) ────────────────────────────────────────
        margin = abs(p - decision_threshold)
        breakdown = {
            "spoof_probability":   round(p, 4),
            "decision_threshold":  round(decision_threshold, 4),
            "margin_from_boundary": round(margin, 4),
            "risk_basis":          "model_probability",
            "policy_engine":       "deterministic",
            "ml_model":            False,
            "thresholds_used": {
                "critical": RiskEngine.CRITICAL_THRESHOLD,
                "high":     RiskEngine.HIGH_THRESHOLD,
                "medium":   RiskEngine.MEDIUM_THRESHOLD,
            },
        }

        return {
            "risk_score":        risk_score,
            "threat_level":      threat_level,
            "recommended_action": action,
            "breakdown":         breakdown,
        }

    @staticmethod
    def _recommended_action(
        threat_level: str,
        spoof_prob: float,
        threshold: float,
    ) -> str:
        margin = abs(spoof_prob - threshold)
        low_confidence = margin < 0.10  # within 10 % of decision boundary

        if threat_level == "CRITICAL":
            return (
                "BLOCK — High-confidence synthetic speech detected. "
                "Reject audio. Log incident. Escalate to security review."
            )
        elif threat_level == "HIGH":
            if low_confidence:
                return (
                    "CHALLENGE — Likely synthetic speech. "
                    "Require secondary verification (e.g. CAPTCHA, callback)."
                )
            return (
                "CHALLENGE — Synthetic speech detected with high confidence. "
                "Reject or require additional authentication."
            )
        elif threat_level == "MEDIUM":
            if low_confidence:
                return (
                    "FLAG — Borderline result near decision boundary. "
                    "Apply additional scrutiny. Manual review recommended."
                )
            return (
                "FLAG — Elevated synthetic speech probability. "
                "Flag for review. Apply additional authentication factor."
            )
        else:  # LOW
            if low_confidence:
                return (
                    "ACCEPT (low confidence) — Classified as bonafide but near "
                    "decision boundary. Monitor for repeated attempts."
                )
            return (
                "ACCEPT — Classified as bonafide with high confidence. "
                "Proceed normally."
            )

    @staticmethod
    def threat_color(threat_level: str) -> str:
        """Return a CSS color class name for the given threat level."""
        return {
            "LOW":      "threat-low",
            "MEDIUM":   "threat-medium",
            "HIGH":     "threat-high",
            "CRITICAL": "threat-critical",
            "UNKNOWN":  "threat-unknown",
        }.get(threat_level, "threat-unknown")

    @staticmethod
    def threat_severity(threat_level: str) -> int:
        """Return numeric severity 0–3 for sorting."""
        return {
            "LOW":      0,
            "MEDIUM":   1,
            "HIGH":     2,
            "CRITICAL": 3,
        }.get(threat_level, -1)


# ── Quick self-test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_cases = [
        (0.02, 0.42),   # very safe bonafide
        (0.35, 0.42),   # borderline bonafide
        (0.50, 0.42),   # borderline spoof
        (0.75, 0.42),   # high spoof
        (0.97, 0.42),   # critical spoof
    ]
    print("Risk Engine Self-Test")
    print("=" * 60)
    for p, thr in test_cases:
        r = RiskEngine.score(p, thr)
        cls = "SPOOF" if p >= thr else "BONA_FIDE"
        print(
            f"  p={p:.2f} thr={thr:.2f} → {cls:10s} | "
            f"risk={r['risk_score']:.3f} | "
            f"{r['threat_level']:8s} | "
            f"{r['recommended_action'][:60]}..."
        )
