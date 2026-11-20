"""
Math Reasoning Micro-Helper
Symbolic derivative/integral/templates and word-problem patterns.
Extended symbolic domains.
"""
import re
import math

def solve_query(query):
    q = query.lower().strip()
    # Normalize word powers to symbolic
    q = re.sub(r"x\s+squared", "x^2", q)
    q = re.sub(r"x\s+cubed", "x^3", q)
    q = re.sub(r"x\s+to the fourth", "x^4", q)
    q = re.sub(r"x\s+to the fifth", "x^5", q)
    # Arithmetic evaluation
    m = re.search(r"what is (\d+)\s*\*\s*(\d+)", q)
    if m:
        return str(int(m.group(1)) * int(m.group(2)))
    m = re.search(r"what is (\d+)\s*\/\s*(\d+)", q)
    if m:
        denom = int(m.group(2))
        return str(int(m.group(1)) // denom if denom else "NaN")
    m = re.search(r"what is (\d+)\s*\+\s*(\d+)", q)
    if m:
        return str(int(m.group(1)) + int(m.group(2)))
    m = re.search(r"what is (\d+)\s*-\s*(\d+)", q)
    if m:
        return str(int(m.group(1)) - int(m.group(2)))
    # Power
    m = re.search(r"what is (\d+)\^(\d+)", q)
    if m:
        return str(int(m.group(1))**int(m.group(2)))

    # Logarithms
    m = re.search(r"log\(10\)\s*of\s*(\d+)", q)
    if m:
        return str(round(math.log10(int(m.group(1))),6))
    m = re.search(r"ln\(?(\d+)\)?", q)
    if m:
        return str(round(math.log(int(m.group(1))),6))
    m = re.search(r"log base 2 of (\d+)", q)
    if m:
        return str(round(math.log2(int(m.group(1))),6))

    # Exponents
    m = re.search(r"e\^(\d+)", q)
    if m:
        return str(round(math.exp(int(m.group(1))),6))

    # Derivatives
    if "derivative of" in q or "d/dx of" in q or "differentiate" in q:
        # basic trig
        if re.search(r"derivative of sin\(x\)", q):
            return "cos(x)"
        if re.search(r"derivative of cos\(x\)", q):
            return "-sin(x)"
        if re.search(r"derivative of tan\(x\)", q):
            return "sec(x)^2"
        if re.search(r"derivative of sec\(x\)", q):
            return "sec(x)tan(x)"
        if re.search(r"derivative of csc\(x\)", q):
            return "-csc(x)cot(x)"
        # ln
        m = re.search(r"derivative of ln\(x\^?(\d*)\)", q)
        if m:
            n = int(m.group(1) or "1")
            return f"{n}/x"
        if re.search(r"derivative of ln\(x\)", q):
            return "1/x"
        # log base10
        if re.search(r"derivative of log\(x\)", q):
            return "1/(x ln(10))"
        # e^x
        if re.search(r"derivative of e\^x", q):
            return "e^x"
        if re.search(r"derivative of e\^\(?(\w+)\)?", q):
            inner = re.search(r"derivative of e\^\(?(\w+)\)?", q)
            if inner:
                return f"e^{inner.group(1)} * d/dx {inner.group(1)}"
        # x^n
        m = re.search(r"derivative of x\^(\d+)", q)
        if m:
            n = int(m.group(1))
            return f"{n}x^{n-1}" if n>1 else "1"
        # coefficient
        m = re.search(r"derivative of (\d+)x\^(\d+)", q)
        if m:
            a = int(m.group(1)); n = int(m.group(2))
            return f"{a*n}x^{n-1}" if n>1 else str(a*n)
        # product rule pattern
        if "product rule" in q:
            return "u'v + u v'"
        # quotient rule
        if "quotient rule" in q:
            return "(u'v - u v')/v^2"
        # chain rule
        if "chain rule" in q:
            return "d/dx f(g(x)) = f'(g(x)) g'(x)"
        # power of sin
        if re.search(r"derivative of sin\^(\d+)\(x\)", q):
            n = int(re.search(r"derivative of sin\^(\d+)\(x\)", q).group(1))
            return f"{n} sin^{n-1}(x) cos(x)"
    # Integrals
    if "integral" in q or "integrate" in q:
        # definite integral simple power
        m = re.search(r"integral from (\d+) to (\d+) of x\^(\d+)", q)
        if m:
            a=int(m.group(1)); b=int(m.group(2)); n=int(m.group(3))
            val = (b**(n+1)-a**(n+1))/(n+1)
            return str(int(val)) if val.is_integer() else str(round(val,6))
        # indefinite
        if re.search(r"(integral of|integrate) e\^x", q):
            return "e^x + C"
        if re.search(r"(integral of|integrate) sin\(x\)", q):
            return "-cos(x) + C"
        if re.search(r"(integral of|integrate) cos\(x\)", q):
            return "sin(x) + C"
        if re.search(r"(integral of|integrate) tan\(x\)", q):
            return "-ln|cos(x)| + C"
        if re.search(r"(integral of|integrate) 1/x", q):
            return "ln|x| + C"
        if re.search(r"(integral of|integrate) ln\(x\)", q):
            return "x ln(x) - x + C"
        m = re.search(r"(integral of|integrate) x\^(\d+)", q)
        if m:
            n = int(m.group(2))
            return f"x^{n+1}/({n+1}) + C"
        m = re.search(r"(integral of|integrate) (\d+)x\^(\d+)", q)
        if m:
            a = int(m.group(1)); n = int(m.group(2))
            denom = n+1
            coeff = a/denom
            if coeff.is_integer():
                coeff = int(coeff)
                return f"{coeff}x^{n+1} + C"
            return f"{coeff}x^{n+1} + C"
    # Limits
    if "limit" in q:
        m = re.search(r"limit x->(\d+) of (\d+)/x", q)
        if m:
            x=int(m.group(1)); num=int(m.group(2))
            return str(num/x if x!=0 else "undefined")
        if re.search(r"limit x->0 of sin\(x\)/x", q):
            return "1"
        if re.search(r"limit x->infinity of 1/x", q):
            return "0"
    # Series summation
    m = re.search(r"sum of first (\d+) natural numbers", q)
    if m:
        n=int(m.group(1))
        return str(n*(n+1)//2)
    m = re.search(r"sum of (\d+) to (\d+)", q)
    if m:
        a=int(m.group(1)); b=int(m.group(2))
        n=b-a+1
        return str((a+b)*n//2)
    # Matrix ops simple 2x2 determinant
    m = re.search(r"det of \[\[(\d+),(\d+)\],\[(\d+),(\d+)\]\]", q)
    if m:
        a,b,c,d = map(int,m.groups())
        return str(a*d - b*c)
    # Partial fractions pattern
    if "partial fractions" in q:
        return "A/(x-a) + B/(x-b)"
    # Trig identities
    if "sin^2(x)+cos^2(x)" in q:
        return "1"
    if "tan(x) = sin(x)/cos(x)" in q:
        return "True"
    # Word problems
    m = re.search(r"if a (?:car|train|vehicle) travels (\d+) km/?h for (\d+) hours", q)
    if m:
        return str(int(m.group(1))*int(m.group(2)))
    m = re.search(r"if a rectangle has length (\d+) and width (\d+)", q)
    if m:
        return str(int(m.group(1))*int(m.group(2)))
    m = re.search(r"what is (\d+)% of (\d+)", q)
    if m:
        return str(int(int(m.group(2))*int(m.group(1))/100))
    m = re.search(r"what is the (\d+)(?:st|nd|rd|th) term of the fibonacci", q)
    if m:
        n = int(m.group(1))
        a,b = 0,1
        for _ in range(n-1):
            a,b = b,a+b
        return str(a)
    # Quadratic solver x^2 - b x + c = 0
    m = re.search(r"solve\s+x\^2\s*-\s*(\d+)x\s*\+\s*(\d+)=0", q)
    if m:
        b = int(m.group(1)); c = int(m.group(2))
        disc = b*b - 4*c
        if disc <0:
            return "no real roots"
        s = int(math.isqrt(disc)) if int(math.isqrt(disc))**2 == disc else None
        if s:
            r1 = (b + s)//2
            r2 = (b - s)//2
            return f"{r1},{r2}"
        return "roots complex"
    # Quadratic general ax^2+bx+c=0
    m = re.search(r"solve\s+(\d+)x\^2\s*\+\s*(-?\d+)x\s*\+\s*(-?\d+)=0", q)
    if m:
        a=int(m.group(1)); b=int(m.group(2)); c=int(m.group(3))
        disc = b*b - 4*a*c
        if disc <0:
            return "no real roots"
        sqrt_disc = math.sqrt(disc)
        r1 = (-b + sqrt_disc)/(2*a)
        r2 = (-b - sqrt_disc)/(2*a)
        return f"{round(r1,4)},{round(r2,4)}"
    # Linear equation ax + b = c
    m = re.search(r"solve\s+(\d+)x\s*\+\s*(-?\d+)\s*=\s*(-?\d+)", q)
    if m:
        a = int(m.group(1)); b = int(m.group(2)); c = int(m.group(3))
        if a != 0:
            x = (c - b)/a
            return str(int(x)) if x.is_integer() else str(x)
    # Trig values
    if q == "sin(pi/2)":
        return "1"
    if q == "cos(pi)":
        return "-1"
    if q == "tan(pi/4)":
        return "1"
    if q == "sin(0)":
        return "0"
    if q == "cos(0)":
        return "1"
    return None

def can_handle(query):
    q = query.lower()
    return any(k in q for k in ["derivative","integral","integrate","limit","sum of","det of","partial fractions","log","ln","e^","travels","rectangle","fibonacci","% of","*","/","^","solve x^2","solve ","what is","sin(","cos(","tan(","matrix","series","logarithm","exponential"])

if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:])
    print(solve_query(q))
