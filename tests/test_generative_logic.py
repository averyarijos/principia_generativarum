import sys
import os
import pytest

# Ensure the workspace root is on sys.path so tests can import the local package
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from . import generative_logic as gl
from generative_logic import Proposition


def deterministic_expansion(contradiction):
    # return a tiny deterministic set so tests are stable
    return {Proposition("alt1"), Proposition("alt2")}


def test_assert_and_no_contradiction():
    engine = gl.GenerativeLogicEngine(expansion_strategy=deterministic_expansion)
    p = Proposition("A")
    state = engine.assert_proposition(p)
    assert state.proposition == p
    assert state.truth_value == gl.GenerativeTruthValue.G_BASIC


def test_contradiction_metabolize():
    engine = gl.GenerativeLogicEngine(expansion_strategy=deterministic_expansion)
    p = Proposition("X")
    engine.assert_proposition(p)
    # assert negation to trigger contradiction
    notp = Proposition(f"¬({p.content})")
    state = engine.assert_proposition(notp)
    # Should have metabolized and created metabolic scar
    assert engine.metrics["contradictions_metabolized"] == 1
    assert len(state.possibility_space) == 2


def test_generative_negation():
    engine = gl.GenerativeLogicEngine()
    p = Proposition("P")
    state = engine.apply_generative_negation(p)
    assert any(prop.content.startswith("¬(") for prop in state.possibility_space)


def test_recursive_enhance_limits():
    engine = gl.GenerativeLogicEngine()
    p = Proposition("R")
    state = engine.recursive_enhance(p, iterations=10)
    # Should saturate at G_TRANSCENDENT
    assert state.truth_value == gl.GenerativeTruthValue.G_TRANSCENDENT


def test_structured_negation_detection():
    engine = gl.GenerativeLogicEngine()
    p = Proposition("Y")
    engine.assert_proposition(p)
    notp = Proposition(f"¬(Y)")
    state = engine.assert_proposition(notp)
    assert engine.metrics["contradictions_metabolized"] == 1


def test_save_and_load(tmp_path):
    engine = gl.GenerativeLogicEngine()
    p = Proposition("S")
    engine.assert_proposition(p)
    path = tmp_path / "state.json"
    engine.save_state(str(path))

    # load back
    loaded = gl.GenerativeLogicEngine.load_state(str(path))
    assert isinstance(loaded, gl.GenerativeLogicEngine)
    assert "S" in loaded.propositions
