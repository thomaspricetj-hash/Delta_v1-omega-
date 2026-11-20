import re, math

def solve_math(expr: str):
    try:
        # allow only safe chars
        if not re.fullmatch(r'[\d\s\.\+\-\*\/\^\%\(\)]+', expr):
            return None
        expr = expr.replace('^','**')
        # safe eval
        return str(eval(expr, {"__builtins__":{}}, {"math":math}))
    except Exception:
        return None
