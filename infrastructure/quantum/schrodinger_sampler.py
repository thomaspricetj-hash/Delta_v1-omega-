"""
Schrödinger-inspired sampling and annealing for TRANSLATION_OS_V1 trainer.

The Schrödinger equation iℏ ∂ψ/∂t = Ĥ ψ is used here as inspiration,
not as a literal PDE solve. We treat a set of candidates as a wave
amplitude ψ and collapse via |ψ|² sampling. Imaginary-time evolution
exp(-Ĥ τ) provides annealing.

This module is intentionally lightweight and CPU-only: it provides
quantum-inspired exploration for window sampling, predictor weighting,
and rotation/LR annealing without changing the core model architecture.
"""

import numpy as np
import math
from typing import List, Sequence

def wavefunction_amplitudes(scores: Sequence[float], temperature: float = 1.0, beta: float = 1.0):
    """
    Map classical scores to wavefunction amplitudes.
    scores -> energy-like values, lower is better.
    amplitudes ∝ exp(-beta * E / T)  -> imaginary-time evolution.
    Returns normalized amplitudes and probabilities |ψ|².
    """
    scores = np.asarray(scores, dtype=np.float64)
    # Shift for numerical stability
    scores -= scores.min()
    # Imaginary-time evolution: ψ ∝ exp(-beta E / T)
    # Avoid overflow
    logits = -beta * scores / max(1e-9, temperature)
    # Softmax normalization for amplitudes
    max_logit = np.max(logits)
    exp_logits = np.exp(logits - max_logit)
    amplitudes = exp_logits / np.sum(exp_logits)
    probs = amplitudes**2
    probs /= probs.sum()
    return amplitudes, probs

def collapse_sample(candidates: List, scores: Sequence[float], temperature: float = 1.0, beta: float = 1.0):
    """
    Quantum-inspired collapse sampling.
    Returns one candidate sampled according to |ψ|².
    """
    _, probs = wavefunction_amplitudes(scores, temperature, beta)
    idx = np.random.choice(len(candidates), p=probs)
    return candidates[idx], idx, probs

def batch_collapse_sample(candidates: List, scores: Sequence[float], k: int, temperature: float = 1.0, beta: float = 1.0, replace=False):
    """
    Sample k candidates without replacement using |ψ|² weights.
    Falls back to uniform if weights are degenerate.
    """
    _, probs = wavefunction_amplitudes(scores, temperature, beta)
    if replace:
        idxs = np.random.choice(len(candidates), size=k, p=probs)
    else:
        # Gumbel-top-k trick for weighted sampling without replacement
        # Use np.random.choice with p if k small relative to N
        if k >= len(candidates):
            idxs = np.arange(len(candidates))
        else:
            idxs = np.random.choice(len(candidates), size=k, p=probs, replace=False)
    return [candidates[i] for i in idxs], idxs, probs

def schrodinger_anneal_schedule(step: int, total_steps: int = 2_000_000, t0: float = 1.0, t_min: float = 0.01):
    """
    Imaginary-time annealing schedule for exploration.
    Temperature decays as T(t) = t_min + (t0 - t_min) * exp(-t / tau)
    Mimics cooling of a quantum system.
    """
    tau = total_steps / 5.0
    temp = t_min + (t0 - t_min) * math.exp(-step / tau)
    beta = 1.0 / max(1e-9, temp)
    return temp, beta

def quantum_rotation_prob(base_prob: float, loss_velocity: float, temperature: float):
    """
    Schrödinger-inspired rotation probability.
    High loss velocity -> higher kinetic energy -> more exploration.
    Modulated by temperature.
    """
    # Kinetic energy proxy
    ke = max(0.0, abs(loss_velocity))
    # Quantum tunneling factor
    tunnel = math.exp(-0.5 / max(1e-9, temperature))
    prob = base_prob * (1.0 + 0.5 * ke) * (0.5 + 0.5 * tunnel)
    return float(min(0.8, max(0.1, prob)))

def quantum_predictor_weight(confidences: Sequence[float], reliabilities: Sequence[float], temperature: float):
    """
    Combine confidence and reliability into a wavefunction amplitude for predictor selection.
    Returns normalized weights for soft masking instead of hard Boruta cut.
    """
    conf = np.asarray(confidences, dtype=np.float64)
    rel = np.asarray(reliabilities, dtype=np.float64)
    # Energy = -log(conf * rel + eps)
    energy = -np.log(np.clip(conf * rel, 1e-9, 1.0))
    _, probs = wavefunction_amplitudes(energy, temperature=temperature, beta=1.0)
    return probs

__all__ = [
    "wavefunction_amplitudes",
    "collapse_sample",
    "batch_collapse_sample",
    "schrodinger_anneal_schedule",
    "quantum_rotation_prob",
    "quantum_predictor_weight",
]
