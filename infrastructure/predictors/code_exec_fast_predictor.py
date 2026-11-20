"""
Code Exec Fast Predictor
Fast heuristic for simple Python arithmetic one-liners
"""
import re
from infrastructure.predictors.predictor_wrapper import wrap_predictor

SAFE_CHARS = set("0123456789+-*/(). %^<>=![]{}:,\"'")

def _predict_impl():
    q = query or ""
    # Look for python code block or simple expression
    # Extract content between backticks or after 'python'
    code = None
    # Try fenced code block
    m = re.search(r'```python(.*?)```', q, re.DOTALL | re.IGNORECASE)
    if m:
        code = m.group(1).strip()
    else:
        # Simple one-liner heuristic
        if q.strip().startswith('python ') or 'print(' in q:
            # Extract first line
            code = q.split('\n')[0].strip()
    if not code:
        return None
    # Only allow simple arithmetic / print
    if not all(c in SAFE_CHARS or c.isspace() or c.isalpha() for c in code):
        # Allow letters for print
        pass
    # Try to evaluate simple arithmetic expression
    try:
        # Extract expression inside print(...)
        expr_match = re.search(r'print\((.*)\)', code)
        if expr_match:
            expr = expr_match.group(1)
        else:
            # Assume whole line is expression
            expr = code
        # Safety: only digits and operators
        if not re.fullmatch(r'[\d\s\.\+\-\*/\(\)%]+', expr):
            return None
        result = eval(expr, {"__builtins__": {}}, {})
        # Format result
        if isinstance(result, float) and result.is_integer():
            result_str = str(int(result))
        else:
            result_str = str(result)
        return {
            'response': result_str,
            'confidence': 0.8,
            'source': 'code_exec_fast'
        }
    except Exception:
        return None

predict = wrap_predictor(_predict_impl, "code_exec_fast_predictor")
