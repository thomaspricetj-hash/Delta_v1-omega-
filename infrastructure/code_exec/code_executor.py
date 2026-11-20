import subprocess, tempfile
from pathlib import Path

class code_executor:
    def __init__(self):
        self.enabled = True

    def execute_code(self, lang, code):
        if not self.enabled or lang != 'python':
            return None, 'Unsupported or disabled'
        try:
            with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as f:
                f.write(code)
                path = f.name
            out = subprocess.check_output(['python', path], stderr=subprocess.STDOUT, timeout=5)
            Path(path).unlink(missing_ok=True)
            return out.decode(errors='ignore'), None
        except subprocess.TimeoutExpired:
            return None, 'Timeout'
        except Exception as e:
            return None, str(e)

    def try_execute_from_query(self, query):
        # Simple heuristic: if query contains code block or python expression
        q = query.lower()
        if '```python' in query or '```py' in query:
            # Extract code block
            import re
            m = re.search(r'```(?:python|py)?\n(.*?)```', query, re.DOTALL)
            if m:
                code = m.group(1)
                out, err = self.execute_code('python', code)
                if err is None:
                    return {"response": out.strip(), "confidence": 0.95, "source": "code_exec"}
        # Simple arithmetic expression detection
        import re
        if re.search(r'\d+\s*[+\-*/]\s*\d+', query):
            # Extract expression
            m = re.search(r'(\d+\s*[+\-*/]\s*\d+)', query)
            if m:
                expr = m.group(1)
                # Safe eval for basic arithmetic
                try:
                    # Very limited eval
                    allowed = set("0123456789+-*/(). ")
                    if all(c in allowed for c in expr):
                        result = eval(expr, {"__builtins__": {}}, {})
                        return {"response": str(result), "confidence": 0.9, "source": "code_exec"}
                except Exception:
                    pass
        return None

def execute_code(lang, code):
    if lang != 'python':
        return None, 'Unsupported language'
    try:
        with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as f:
            f.write(code)
            path = f.name
        out = subprocess.check_output(['python', path], stderr=subprocess.STDOUT, timeout=5)
        Path(path).unlink(missing_ok=True)
        return out.decode(errors='ignore'), None
    except subprocess.TimeoutExpired:
        return None, 'Timeout'
    except Exception as e:
        return None, str(e)
