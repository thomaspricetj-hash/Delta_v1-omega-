"""
Uncertainty predictor — estimates confidence before full reasoning
"""
def _predict_impl():
    # Simple heuristic: short queries or unknown entities -> high uncertainty
    q_low = query.lower()
    # If query contains 'what is' and is short, assume medium uncertainty
    if len(q_low.split()) < 6:
        # Trigger deeper reasoning path
        return {'source':'uncertainty_predictor','response':None,'confidence':0.3,'needs_deeper':True}
    return None

predict = wrap_predictor(_predict_impl, "uncertainty_predictor")
