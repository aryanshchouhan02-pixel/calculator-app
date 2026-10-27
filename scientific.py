import ast
import math
import operator
import re
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()


class CalcError(Exception):
    pass


FUNCS1 = {"sin", "cos", "tan", "asin", "acos", "atan",
          "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
          "log", "ln", "log2", "sqrt", "cbrt", "abs", "exp", "fact",
          "floor", "ceil", "round", "sign", "todeg", "torad", "ang"}
FUNCS2 = {"ncr", "npr", "gcd", "lcm"}
FUNCS = FUNCS1 | FUNCS2
CONSTS = {"pi", "e", "ans"}
POSTFIX = {"!": "fact", "\u00b0": "ang"}
TOKEN_RE = re.compile(r"\d+\.?\d*|\.\d+|[a-z]+[0-9]*|\*\*|[-+*/%(),!\u00b0]")
BIN = {ast.Add: operator.add, ast.Sub: operator.sub,
       ast.Mult: operator.mul, ast.Div: operator.truediv,
       ast.Mod: operator.mod}
SIMPLE = {
    "log": math.log10, "ln": math.log, "log2": math.log2,
    "sqrt": math.sqrt,
    "cbrt": lambda v: math.copysign(abs(v) ** (1 / 3), v),
    "abs": abs, "exp": math.exp,
    "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh,
    "asinh": math.asinh, "acosh": math.acosh, "atanh": math.atanh,
    "floor": math.floor, "ceil": math.ceil,
    "round": lambda v: math.floor(v + 0.5),
    "sign": lambda v: (v > 0) - (v < 0),
}


def tokenize(expr):
    s = expr.lower()
    for a, b in (("\u00d7", "*"), ("\u00f7", "/"), ("\u2212", "-"),
                 ("\u03c0", "pi"), ("\u221a", "sqrt"), ("^", "**")):
        s = s.replace(a, b)
    s = re.sub(r"\bmod\b", "%", s)
    raw = TOKEN_RE.findall(s)
    if "".join(raw) != re.sub(r"\s+", "", s):
        raise CalcError("Invalid character in expression")
    tokens = []
    for t in raw:
        m = re.fullmatch(r"([a-z]+)(\d+)", t)
        if (m and t not in FUNCS and t not in CONSTS
                and (m.group(1) in FUNCS or m.group(1) in CONSTS)):
            tokens += [m.group(1), m.group(2)]
        else:
            tokens.append(t)
    for t in tokens:
        if t[0].isalpha() and t not in FUNCS and t not in CONSTS:
            raise CalcError("Unknown name: " + t)
    return tokens


def kind(t):
    if t[0].isdigit() or t[0] == ".":
        return "num"
    if t in CONSTS:
        return "const"
    if t in FUNCS:
        return "func"
    return t if t in ("(", ")") else "op"


def wrap_bare_functions(tokens):
    """sin 30 -> sin(30)"""
    out, i = [], 0
    while i < len(tokens):
        t = tokens[i]
        out.append(t)
        i += 1
        if t in FUNCS and i < len(tokens) and tokens[i] != "(":
            j = i
            if tokens[j] in ("-", "+"):
                j += 1
            if j < len(tokens) and kind(tokens[j]) in ("num", "const"):
                while j + 1 < len(tokens) and tokens[j + 1] in POSTFIX:
                    j += 1
                out.append("(")
                out.extend(tokens[i:j + 1])
                out.append(")")
                i = j + 1
            else:
                raise CalcError("Missing bracket after " + t)
    return out


def expand_postfix(tokens):
    """5! -> fact(5), 30° -> ang(30)"""
    out = []
    for t in tokens:
        if t not in POSTFIX:
            out.append(t)
            continue
        name = POSTFIX[t]
        if not out:
            raise CalcError("Invalid use of " + t)
        if out[-1] == ")":
            depth, i = 0, len(out) - 1
            while i >= 0:
                if out[i] == ")":
                    depth += 1
                elif out[i] == "(":
                    depth -= 1
                    if depth == 0:
                        break
                i -= 1
            if i < 0:
                raise CalcError("Brackets don't match")
            if i > 0 and out[i - 1] in FUNCS:
                i -= 1
            out[i:] = [name, "("] + out[i:] + [")"]
        elif kind(out[-1]) in ("num", "const"):
            out[-1:] = [name, "(", out[-1], ")"]
        else:
            raise CalcError("Invalid use of " + t)
    return out


def add_implicit_multiplication(tokens):
    out = []
    for t in tokens:
        if (out and kind(out[-1]) in ("num", "const", ")")
                and kind(t) in ("num", "const", "func", "(")):
            out.append("*")
        out.append(t)
    return out


def _int(x):
    if x != int(x) or abs(x) > 1e12:
        raise ValueError
    return int(x)


def _call(name, a, deg):
    x = a[0]
    if name in ("sin", "cos", "tan"):
        if deg:
            x = math.radians(x)
        if name == "tan" and abs(math.cos(x)) < 1e-12:
            raise ValueError
        return getattr(math, name)(x)
    if name in ("asin", "acos", "atan"):
        r = getattr(math, name)(x)
        return math.degrees(r) if deg else r
    if name == "ang":
        return x if deg else math.radians(x)
    if name == "todeg":
        return math.degrees(x)
    if name == "torad":
        return math.radians(x)
    if name == "fact":
        if x < 0 or x > 170 or x != int(x):
            raise ValueError
        return float(math.factorial(int(x)))
    if name in FUNCS2:
        n, k = _int(a[0]), _int(a[1])
        if name in ("ncr", "npr"):
            if n < 0 or k < 0 or n > 1000:
                raise ValueError
            return float(math.comb(n, k) if name == "ncr" else math.perm(n, k))
        return float(math.gcd(n, k) if name == "gcd" else math.lcm(n, k))
    return float(SIMPLE[name](x))


def _eval(node, deg, ans):
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return float(node.value)
    if isinstance(node, ast.Name):
        return {"pi": math.pi, "e": math.e, "ans": ans}[node.id]
    if isinstance(node, ast.UnaryOp) and type(node.op) in (ast.UAdd, ast.USub):
        v = _eval(node.operand, deg, ans)
        return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.BinOp):
        left = _eval(node.left, deg, ans)
        right = _eval(node.right, deg, ans)
        if isinstance(node.op, ast.Pow):
            if abs(right) > 10000:
                raise OverflowError
            return left ** right
        if type(node.op) in BIN:
            return BIN[type(node.op)](left, right)
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in FUNCS and not node.keywords):
        name = node.func.id
        need = 2 if name in FUNCS2 else 1
        if len(node.args) != need:
            raise CalcError("Wrong number of values for " + name)
        return _call(name, [_eval(x, deg, ans) for x in node.args], deg)
    raise CalcError("Invalid expression")


def evaluate(expr, mode="deg", ans=0.0):
    if len(expr) > 200:
        raise CalcError("Expression too long")
    tokens = add_implicit_multiplication(
        expand_postfix(wrap_bare_functions(tokenize(expr))))
    try:
        tree = ast.parse("".join(tokens), mode="eval")
        value = _eval(tree.body, mode == "deg", ans)
    except ZeroDivisionError:
        raise CalcError("Division by zero")
    except OverflowError:
        raise CalcError("Number too large")
    except ValueError:
        raise CalcError("Math error")
    except (SyntaxError, RecursionError, TypeError, KeyError):
        raise CalcError("Invalid expression")
    if isinstance(value, complex) or not math.isfinite(value):
        raise CalcError("Math error")
    return round(value, 10) + 0.0


class CalcRequest(BaseModel):
    expr: str = Field(max_length=200)
    mode: Literal["deg", "rad"] = "deg"
    ans: float = 0.0


@router.post("/scientific")
def scientific(body: CalcRequest):
    try:
        return {"result": evaluate(body.expr, body.mode, body.ans)}
    except CalcError as e:
        raise HTTPException(status_code=400, detail=str(e))