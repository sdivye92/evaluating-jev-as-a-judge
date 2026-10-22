"""Execution-based labels for MBPP.

Gold solutions are verified against the unit tests, then mutated with a single AST edit.
A mutant is kept as an INCORRECT candidate only if it still runs but fails a test with an
AssertionError (a plausible-looking wrong program, not a syntax or name error).

This runs dataset code in a subprocess with CPU/memory limits. It is not a security
sandbox; do not point it at untrusted code.
"""

from __future__ import annotations

import ast
import random
import resource
import subprocess
import sys
import tempfile
from dataclasses import dataclass

TIMEOUT_S = 5


def _limits() -> None:  # runs in the child process before exec
    resource.setrlimit(resource.RLIMIT_CPU, (TIMEOUT_S, TIMEOUT_S))
    resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))


@dataclass
class ExecResult:
    passed: bool
    assertion_failed: bool
    timed_out: bool
    stderr: str


def run_tests(code: str, test_imports: list[str], tests: list[str]) -> ExecResult:
    program = "\n".join([*test_imports, code, *tests]) + "\n"
    with tempfile.TemporaryDirectory() as tmp:
        try:
            p = subprocess.run(
                [sys.executable, "-I", "-c", program],
                capture_output=True,
                text=True,
                timeout=TIMEOUT_S + 2,
                cwd=tmp,
                env={},
                preexec_fn=_limits,
            )
        except subprocess.TimeoutExpired:
            return ExecResult(False, False, True, "timeout")
    return ExecResult(
        passed=p.returncode == 0,
        assertion_failed="AssertionError" in p.stderr,
        timed_out=False,
        stderr=p.stderr[-500:],
    )


# ------------------------------------------------------------------------- mutation

_CMP = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt, ast.Eq: ast.NotEq, ast.NotEq: ast.Eq}
_ARITH = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.Add, ast.FloorDiv: ast.Mult, ast.Mod: ast.FloorDiv}
_BOOL = {ast.And: ast.Or, ast.Or: ast.And}


def normalize(code: str) -> str:
    """Round-trip through the AST so correct and mutated code share formatting (no comments)."""
    return ast.unparse(ast.parse(code))


def _sites(tree: ast.AST) -> list[ast.AST]:
    out: list[ast.AST] = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Compare) and type(n.ops[0]) in _CMP:
            out.append(n)
        elif isinstance(n, ast.BinOp) and type(n.op) in _ARITH:
            out.append(n)
        elif isinstance(n, ast.BoolOp) and type(n.op) in _BOOL:
            out.append(n)
        elif isinstance(n, ast.Constant) and type(n.value) is int:
            out.append(n)
    return out


def mutate(code: str, rng: random.Random) -> str | None:
    tree = ast.parse(code)
    sites = _sites(tree)
    if not sites:
        return None
    target = rng.choice(sites)  # the node is part of `tree`, so editing it edits the tree
    if isinstance(target, ast.Compare):
        target.ops[0] = _CMP[type(target.ops[0])]()
    elif isinstance(target, ast.BinOp):
        target.op = _ARITH[type(target.op)]()
    elif isinstance(target, ast.BoolOp):
        target.op = _BOOL[type(target.op)]()
    else:
        target.value = target.value + rng.choice([-1, 1])
    return ast.unparse(tree)


def make_incorrect(
    code: str, test_imports: list[str], tests: list[str], rng: random.Random, attempts: int = 12
) -> tuple[str, ExecResult] | None:
    """Return a mutant that fails a test with AssertionError, or None."""
    for _ in range(attempts):
        mutant = mutate(code, rng)
        if mutant is None:
            return None
        if mutant == code:
            continue
        res = run_tests(mutant, test_imports, tests)
        if not res.passed and res.assertion_failed and not res.timed_out:
            return mutant, res
    return None
