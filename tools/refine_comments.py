"""Refine EOL comments in generative_logic_commented.py

Removes generic '  # auto: code' markers and adds richer end-of-line comments
for top-level class and def declarations only.
"""
from pathlib import Path

SRC = Path("/workspaces/projects/PrincipiaGenerativarum/generativity/generative_logic_commented.py")

MAPPING = {
    "def now_utc()": "  # Helper: return timezone-aware UTC datetime for timestamps",
    "class GenerativeTruthValue(": "  # Multi-valued generative truth (G_HINGE..G_TRANSCENDENT)",
    "class Proposition(": "  # Immutable proposition object (content + metadata)",
    "class GenerativeState(": "  # State container: proposition, truth value, possibility space, trace",
    "class Contradiction(": "  # Represents a detected contradiction with context and timestamp",
    "class MetabolicScar(": "  # Persistent record of a metabolized contradiction (scar)",
    "class LogicalOperator(": "  # Abstract base class for logical operators",
    "class GenerativeZeroOperator(": "  # Operator metabolizing contradictions into possibility sets (⊙₀)",
    "class GenerativeNegationOperator(": "  # Operator implementing generative negation (∇_g)",
    "class MetabolicCompositionOperator(": "  # Operator for metabolic composition (⊗)",
    "class GenerativeLogicEngine(": "  # Core engine: asserts, detects contradictions, metabolizes, metrics",
    "class GenerativeLogicMetricsCollector(": "  # Exposes engine metrics in a Prometheus-like format",
    "def main(": "  # Demo runner for the module"
}

text = SRC.read_text(encoding="utf-8")
lines = text.splitlines()
out = []
for line in lines:
    # remove generic marker if present
    if line.rstrip().endswith("# auto: code"):
        line = line.replace("# auto: code", "").rstrip()
    # only modify top-level declarations
    stripped = line.lstrip()
    indent = len(line) - len(stripped)
    if indent == 0:
        for key, comment in MAPPING.items():
            if stripped.startswith(key):
                # ensure we don't duplicate comment
                if not line.endswith(comment):
                    line = line + comment
                break
    out.append(line)

SRC.write_text("\n".join(out) + "\n", encoding="utf-8")
print("Refined comments in:", SRC)
