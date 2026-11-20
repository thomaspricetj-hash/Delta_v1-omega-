"""
Cognitive Gyrotron Layer
Idea Generation → Resonance Chamber → Signal Amplifier → Mode Selector → Coherence Validator → Response
Implements magnetic confinement, standing waves, resonance graph
"""
import time
from collections import defaultdict

class ResonanceChamber:
    def __init__(self):
        self.pass_count = 3

    def resonate(self, ideas):
        # Find overlap / clustering
        # Simple overlap by token intersection
        scores = {}
        for i, a in enumerate(ideas):
            for j, b in enumerate(ideas[i+1:], i+1):
                overlap = len(set(a.lower().split()) & set(b.lower().split()))
                scores.setdefault(a, 0)
                scores.setdefault(b, 0)
                scores[a] += overlap
                scores[b] += overlap
        # Return ideas with overlap weight
        return sorted(ideas, key=lambda x: scores.get(x,0), reverse=True)

class SignalAmplifier:
    def amplify(self, ideas, context):
        # score = confidence * historical_success * relevance * consensus
        scored = []
        for idea in ideas:
            confidence = context.get('confidence', 0.5)
            hist = context.get('historical_success', 0.5)
            relevance = context.get('relevance', 0.5)
            consensus = context.get('consensus', 0.5)
            score = confidence * hist * relevance * consensus
            scored.append((idea, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

class ModeSelector:
    MODES = ['Knowledge','Reasoning','Planning','Coding','Creative']
    def select(self, query):
        q = query.lower()
        if any(k in q for k in ['code','function','def ', 'class ']):
            return 'Coding'
        if any(k in q for k in ['plan','step','how to']):
            return 'Planning'
        if any(k in q for k in ['derive','integrate','prove','why']):
            return 'Reasoning'
        if any(k in q for k in ['write','story','poem','creative']):
            return 'Creative'
        return 'Knowledge'

class MagneticConfinement:
    def confine(self, query, candidates):
        # Constrain reasoning to problem map / decision tree / knowledge graph
        # Simple filter: keep candidates with keyword overlap to query
        qwords = set(query.lower().split())
        filtered = []
        for cand in candidates:
            cwords = set(str(cand).lower().split())
            if qwords & cwords:
                filtered.append(cand)
        return filtered or candidates

class StandingWaves:
    def __init__(self):
        self.patterns = defaultdict(int)

    def record(self, problem_type, solution_path):
        key = f"{problem_type}|{solution_path}"
        self.patterns[key] += 1

    def best_pattern(self, problem_type):
        best = None
        best_count = 0
        for k, cnt in self.patterns.items():
            if k.startswith(problem_type+'|'):
                if cnt > best_count:
                    best_count = cnt
                    best = k.split('|',1)[1]
        return best

class ResonanceGraph:
    def __init__(self):
        self.edges = defaultdict(int)

    def reinforce(self, ideas):
        for i in range(len(ideas)):
            for j in range(i+1, len(ideas)):
                pair = tuple(sorted([ideas[i], ideas[j]]))
                self.edges[pair] += 1

    def strong_clusters(self, threshold=3):
        return [pair for pair, w in self.edges.items() if w >= threshold]

class CognitiveGyrotron:
    def __init__(self):
        self.chamber = ResonanceChamber()
        self.amplifier = SignalAmplifier()
        self.mode_selector = ModeSelector()
        self.confinement = MagneticConfinement()
        self.standing_waves = StandingWaves()
        self.graph = ResonanceGraph()

    def process(self, query, idea_candidates, context=None):
        context = context or {}
        mode = self.mode_selector.select(query)
        # Magnetic confinement
        confined = self.confinement.confine(query, idea_candidates)
        # Resonance chamber
        resonated = self.chamber.resonate(confined)
        # Signal amplifier
        amplified = self.amplifier.amplify(resonated, context)
        # Standing wave pattern reuse
        problem_type = mode
        best_wave = self.standing_waves.best_pattern(problem_type)
        # Resonance graph reinforcement
        self.graph.reinforce(resonated)
        # Coherence validator - pick top amplified idea
        top_idea = amplified[0][0] if amplified else None
        # Record standing wave
        if top_idea:
            self.standing_waves.record(problem_type, top_idea)
        return {
            'mode': mode,
            'top_idea': top_idea,
            'amplified': amplified,
            'resonated': resonated,
            'best_wave': best_wave,
            'graph_clusters': self.graph.strong_clusters()
        }

# Singleton
_gyrotron = CognitiveGyrotron()

def get_cognitive_gyrotron():
    return _gyrotron

# Compatibility alias for orchestrator import
cognitive_gyrotron = _gyrotron
