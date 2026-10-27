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


FUNCS = {"sin", "cos", "tan", "asin", "acos", "atan",
         "log", "ln", "sqrt", "abs", "exp", "fact"}
CONSTS = {"pi", "e", "ans"}
TOKEN_RE = re.compile(r"\d+\.?\d*|\.\d+|[a-z]+|\*\*|[-+*/%()!]")
BIN = {ast.Add: operator.add, ast.Sub: operator.sub,
       ast.Mult: operator.mul, ast.Div: operator.truediv,
       ast.Mod: operator.mod}


def tokenize(expr):
    s = expr.lower()
    for a, b in (("\u00d7", "*"), ("\u00f7", "/"), ("\u2212", "-"),
                 ("\u03c0", "pi"), ("\u221a", "sqrt"), ("^", "**")):
        s = s.replace(a, b)
    s = re.sub(r"\bmod\b", "%", s)
    s = re.sub(r"\s+", "", s)
    tokens = TOKEN_RE.findall(s)
    if "".join(tokens) != s:
        raise CalcError("Invalid character in expression")
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


def expand_factorials(tokens):
    out = []
    for t in tokens:
        if t != "!":
            out.append(t)
            continue
        if not out:
            raise CalcError("Invalid use of !")
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
            out[i:] = ["fact", "("] + out[i:] + [")"]
        elif kind(out[-1]) in ("num", "const"):
            out[-1:] = ["fact", "(", out[-1], ")"]
        else:
            raise CalcError("Invalid use of !")
    return out


def add_implicit_multiplication(tokens):
    out = []
    for t in tokens:
        if (out and kind(out[-1]) in ("num", "const", ")")
                and kind(t) in ("num", "const", "func", "(")):
            out.append("*")
        out.append(t)
    return out


def _call(name, x, deg):
    if name in ("sin", "cos", "tan"):
        if deg:
            if name == "tan" and x % 180 == 90:
                raise ValueError
            x = math.radians(x)
        return getattr(math, name)(x)
    if name in ("asin", "acos", "atan"):
        r = getattr(math, name)(x)
        return math.degrees(r) if deg else r
    if name == "fact":
        if x < 0 or x > 170 or x != int(x):
            raise ValueError
        return float(math.factorial(int(x)))
    return {"log": math.log10, "ln": math.log, "sqrt": math.sqrt,
            "abs": abs, "exp": math.exp}[name](x)


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
            and node.func.id in FUNCS and len(node.args) == 1
            and not node.keywords):
        return _call(node.func.id, _eval(node.args[0], deg, ans), deg)
    raise CalcError("Invalid expression")


def evaluate(expr, mode="deg", ans=0.0):
    if len(expr) > 200:
        raise CalcError("Expression too long")
    tokens = add_implicit_multiplication(expand_factorials(tokenize(expr)))
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