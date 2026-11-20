"""
Braided Prediction Merger
Combines multiple predictor outputs with weighted voting
"""
def braid_predictions(predictions):
    # predictions: list of dicts with source, response, confidence
    if not predictions:
        return None
    # Weighted vote by confidence
    scores = {}
    for p in predictions:
        resp = p.get('response')
        conf = p.get('confidence', 0.5)
        scores[resp] = scores.get(resp, 0) + conf
    best_resp = max(scores.items(), key=lambda x: x[1])[0]
    best_conf = scores[best_resp] / sum(scores.values())
    return {
        'source': 'braided_predictor',
        'response': best_resp,
        'confidence': best_conf,
        'braided_scores': scores
    }
