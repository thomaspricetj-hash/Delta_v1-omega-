"""
Focused context attention process for Translation OS V1
Steps:
1. Extract keywords
2. Search all memory sources
3. Score matches with attention_score formula
4. Sort, keep top 20
5. Return Focused Context Bundle
"""
import re
from pathlib import Path
import sys

def extract_keywords(query):
    # simple keyword extraction: alphanumeric tokens, lowercased, length>2
    tokens = re.findall(r'\b\w+\b', query.lower())
    # filter stopwords
    stop = {'what','is','the','a','an','how','why','who','when','where','of','in','on','for','to','and','or','with','by','from','are','be','can','do','does'}
    keywords = [t for t in tokens if len(t)>2 and t not in stop]
    # deduplicate preserve order
    seen=set()
    out=[]
    for k in keywords:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out

def _now_ts():
    import time
    return time.time()

def score_entry(keyword_match, semantic_match, recency, importance_weight):
    return (
        keyword_match * 0.40 +
        semantic_match * 0.30 +
        recency * 0.15 +
        importance_weight * 0.15
    )

def build_focused_context_bundle(query, top_k=20):
    BASE = Path(__file__).resolve().parent.parent.parent.parent
    # ensure paths
    sys.path.insert(0, str(BASE / "runtime"))
    sys.path.insert(0, str(BASE))
    
    keywords = extract_keywords(query)
    # Resonance boost from True Hybrid ResonanceMemory
    resonance_boost = 0.0
    try:
        import torch
        res_path = BASE / "runtime" / "logs" / "resonance_memory.pt"
        if res_path.exists():
            mem = torch.load(res_path, map_location='cpu')
            # simple boost based on memory norm
            resonance_boost = float(torch.norm(mem).item()) / 1000.0
            resonance_boost = min(1.0, resonance_boost)
    except Exception:
        resonance_boost = 0.0
    results = []
    
    # Helper to normalize score to 0-1
    def norm(v, max_v=1.0):
        try:
            return max(0.0, min(1.0, float(v)/max_v))
        except:
            return 0.0
    
    # 1. Semantic map
    try:
        from semantic_map import predict_semantic
        sem = predict_semantic(query, top_k=50, threshold=0, min_words=1)
        for c in sem.get('candidates', []):
            ans = c.get('answer','')
            score_sem = c.get('score',0.0)
            # keyword match
            km = sum(1 for k in keywords if k in ans.lower() or k in query.lower()) / max(1, len(keywords))
            recency = 0.5  # placeholder
            importance = 0.7 + resonance_boost
            att = score_entry(km, norm(score_sem, 20.0), recency, importance)
            results.append({
                'source':'semantic_map',
                'answer': ans,
                'keyword_match': km,
                'semantic_match': norm(score_sem,20.0),
                'recency': recency,
                'importance_weight': importance,
                'attention_score': att
            })
    except Exception:
        pass
    
    # 2. Pattern mapper
    try:
        from pattern_mapper import get_pattern_candidates
        pats = get_pattern_candidates(query, 50)
        for p in pats:
            ans = p.get('answer','')
            score_pat = p.get('score',0.0)
            km = sum(1 for k in keywords if k in ans.lower() or k in query.lower()) / max(1, len(keywords))
            recency = 0.5
            importance = 0.6 + resonance_boost
            att = score_entry(km, norm(score_pat,20.0), recency, importance)
            results.append({
                'source':'pattern_mapper',
                'answer': ans,
                'keyword_match': km,
                'semantic_match': norm(score_pat,20.0),
                'recency': recency,
                'importance_weight': importance,
                'attention_score': att
            })
    except Exception:
        pass
    
    # 3. Next word map
    try:
        from next_word_map import get_next_probabilities
        words = re.findall(r'\b\w+\b', query.lower())
        if words:
            last = words[-1]
            probs = get_next_probabilities(last, 50)
            for p in probs:
                ans = p.get('word','')
                reward = p.get('reward',0.0)
                km = 1.0 if last in keywords else 0.0
                importance = 0.5 + resonance_boost
                att = score_entry(km, norm(reward,5.0), 0.5, importance)
                results.append({
                    'source':'next_word_map',
                    'answer': ans,
                    'keyword_match': km,
                    'semantic_match': norm(reward,5.0),
                    'recency':0.5,
                    'importance_weight':importance,
                    'attention_score': att
                })
    except Exception:
        pass
    
    # 4. Statistical word relations
    try:
        from statistical_word_relations import get_relations
        for k in keywords[:3]:
            rels = get_relations(k, 20)
            for r in rels:
                ans = r.get('word','')
                pmi = r.get('pmi',0.0)
                km = 1.0 if k in ans.lower() else 0.0
                importance = 0.5 + resonance_boost
                att = score_entry(km, norm(pmi,10.0), 0.5, importance)
                results.append({
                    'source':'statistical_word_relations',
                    'answer': ans,
                    'keyword_match': km,
                    'semantic_match': norm(pmi,10.0),
                    'recency':0.5,
                    'importance_weight':importance,
                    'attention_score': att
                })
    except Exception:
        pass
    
    # Sort and keep top
    results.sort(key=lambda x: x['attention_score'], reverse=True)
    top = results[:top_k]
    bundle = {
        'query': query,
        'keywords': keywords,
        'total_candidates': len(results),
        'top_k': top_k,
        'focused_context': top,
        'attention_summary': {
            'mean_score': sum(r['attention_score'] for r in top)/max(1,len(top)),
            'sources': {}
        }
    }
    # summary by source
    from collections import Counter
    src_counts = Counter(r['source'] for r in top)
    bundle['attention_summary']['sources'] = dict(src_counts)
    return bundle
