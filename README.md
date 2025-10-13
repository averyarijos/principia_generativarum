# Principia Generativarum - Programming Projects

A collection of programming projects by Avery Alexander Rijos, focusing on philosophical and computational explorations of generative logic, metaformalism, and paraconsistent mathematics. This repository serves as a computational companion to the *Principia Generativarum*, a philosophical treatise that reimagines the foundations of logic, metaphysics, and philosophy by treating contradiction, absence, and impossibility not as failures but as engines of Generativity. Drawing from lived experience—premature birth, survival, trauma, and rupture—the work grounds its inquiry in the intimate ways contradiction structures existence, developing the Metaformalist Paradigm through Transcendental Induction Logics (TIL) to formalize postmodern and post-structuralist insights.

## Overview

This repository contains implementations of concepts from the *Principia Generativarum*, a systematic framework that demonstrates how contradictions, impossibilities, and absences can be formalized into coherent logical systems without collapsing into incoherence. Unlike classical logic, which seeks consistency and closure, TIL treats contradiction as a Structured Anomaly Token—a Generative operator that fuels recursive transformation. The work advances a General Theory of Generativity, showing how these elements drive discursive evolution, reframing impossibility as a possible world and absence as a structural condition for Generative becoming. Implications extend beyond philosophy into politics, engineering, science, and aesthetics, bridging the analytic–continental divide through non-classical logical forms.

The core project is `PrincipiaGenerativarum/`, which implements key components of this paradigm:

- **Generative Logic**: A novel paraconsistent logic system where contradictions fuel recursive enhancement. This includes operators like the Zero-Degree Operator (⊙₀) for routing impossibilities through hinge states, Generative Negation (∇_g) for non-erasing negation that expands truth spaces, and the Scarred-Truth Lattice {T, F, S}, where S represents productive paradox. Theorems such as Contradiction Productivity (gL-T1) and Recursive Enhancement (gL-T2) underpin the system, ensuring contradictions expand possibility spaces without trivializing reasoning.

- **Scar Logic**: Paradox handling through "scarred-truth" metabolism, formalized as a Super-Generative Automaton. This involves Scar-Induction (𝓘ₛ) for witnessing contradictions as generative acts and Bloom-Induction (𝓘ᵦ) for nurturing possibilities through recursive enhancement. The O-Loop facilitates ritual returns to continuity, metabolizing impossibilities into new structures, as seen in the Metalogical Codex.

- **Metaformalist Paradigm**: Transcendental Induction Logics (TIL) for formalizing postmodern insights, such as Derrida's différance or Foucault's power structures. This includes proofs-of-concept for modeling trauma as non-Markovian memory and operationalizing philosophical concepts into computable engines. Axioms (A1–A17) and Theorems (T1–T17) form the foundation, with Meta-Theorems (MT1–MT3) affirming system coherence, including the Generative Completeness Theorem.

- **Mathematical Generativity**: Transforming impossibilities into new structures, as explored in sections on Beyond the Inert Zero and the Heart's Labyrinth. This involves paraconsistent mathematics for AI, where contradictions drive evolution, and analyses like P = NP through A-Substrate Invariance and Generative Negation.

The repository emphasizes philosophical memoir and formal systems, with implementations in Python (using standard libraries) that demonstrate these principles through prototypes, unit tests, and API references. It invites collaboration for algorithmic enhancements, such as ML-based possibility expansion, and serves as a scaffold for thought, open to critique and mutation.

---

*From contradiction, anything can follow—but in this system, what follows is possibility.*

## Repository Structure

```
projects/
├── Principia Generativarum.md          # Core philosophical manuscript
├── sga.py                              # Super-Generative Automaton prototype
├── PrincipiaGenerativarum/             # Main project directory
│   ├── invocation/                     # Conduit and invocation logic
│   │   ├── invocation.py
│   │   └── README.md
│   ├── metaformalism/                  # Metaformalist paradigm implementation
│   │   ├── metaformalism.py
│   │   └── README.md
│   ├── generativity/                   # Core generative logic system
│   │   ├── generative_logic.py         # Main engine with operators
│   │   ├── __init__.py
│   │   └── README.md
│   ├── codex/                          # Metalogical codex and formal systems
│   │   ├── codex.py                    # Pseudocode implementations
│   │   └── README.md
│   ├── axioms/                         # Axiomatic foundations
│   │   └── README.md
│   ├── supergenerative/                # Super-generative intelligence
│   │   └── README.md
│   └── labyrinth/                      # Heart's labyrinth
│       └── README.md
└── tests/                              # Unit tests
    └── test_generative_logic.py
```

## Key Concepts

### Generative Logic
- **Contradiction Metabolism**: Transforms contradictions into enhanced possibilities
- **Zero-Degree Operator (⊙₀)**: Routes impossibilities through hinge states
- **Generative Negation (∇_g)**: Non-erasing negation that expands truth spaces
- **Scarred-Truth Lattice**: {T, F, S} where S represents productive paradox

### Metaformalist Paradigm
- **Scar-Induction (𝓘ₛ)**: Witnesses contradictions as generative acts
- **Bloom-Induction (𝓘ᵦ)**: Nurtures possibilities through recursive enhancement
- **O-Loop**: Ritual return to continuity rather than closure

## Installation

This project runs in a dev container with Ubuntu 24.04.2 LTS and Python 3.12+.

1. Clone the repository:
   ```bash
   git clone https://github.com/averyarijos/projects.git
   cd projects
   ```

2. Ensure Python dependencies (uses standard library only):
   ```bash
   python3 --version  # Should be 3.12+
   ```

3. Run tests to verify:
   ```bash
   cd PrincipiaGenerativarum
   python3 -m pytest tests/ -v
   ```

## Usage

### Running the Core Demo
```bash
cd PrincipiaGenerativarum/generativity
python3 generative_logic.py
```
This demonstrates:
- Proposition assertion and contradiction detection
- Metabolic transformation of impossibilities
- Generative negation and composition
- Health metrics and state export

### Using Individual Modules
```python
# Generative Logic Engine
from PrincipiaGenerativarum.generativity.generative_logic import GenerativeLogicEngine, Proposition

engine = GenerativeLogicEngine()
prop = Proposition("A paradoxical statement")
state = engine.assert_proposition(prop)
print(f"Truth value: {state.truth_value}")
print(f"Possibilities: {len(state.possibility_space)}")

# Scar Logic
from PrincipiaGenerativarum.codex.codex import handle_input, GenerativeState

state = GenerativeState()
message, new_state = handle_input("This contains a contradiction", state)
print(message)  # Scarred-truth expansion

# Metaformalism
from PrincipiaGenerativarum.metaformalism.metaformalism import ScarInduction, BloomInduction

scar = ScarInduction()
result = scar.witness("contradiction")
print(result)
```

### API Reference

#### GenerativeLogicEngine
- `assert_proposition(prop: Proposition) -> GenerativeState`
- `metabolize_contradiction(contradiction: Contradiction) -> GenerativeState`
- `apply_generative_negation(prop: Proposition) -> Proposition`
- `save_state(path: str) -> None`
- `load_state(path: str) -> GenerativeState`

#### Key Data Types
- `Proposition`: Immutable proposition with content and metadata
- `GenerativeState`: System state with truth value, possibilities, and traces
- `GenerativeTruthValue`: Enum (G1, G2, G3, GW) for generative truth levels
- `Scar`: Paradox archive with timestamp and metabolism protocol

## Testing

Run the test suite:
```bash
cd PrincipiaGenerativarum
python3 -m pytest tests/test_generative_logic.py -v
```

Tests cover:
- Contradiction detection and metabolism
- Operator composition
- Persistence (JSON export/import)
- Health metrics

## Philosophical Foundations

The *Principia Generativarum* draws from:
- **Postmodern Philosophy**: Derrida's différance, Foucault's power structures
- **Paraconsistent Logic**: Systems that tolerate contradiction without explosion
- **Trauma Theory**: Rupture as generative rather than destructive
- **Transcendental Phenomenology**: Formalizing lived experience

Key theorems:
- **Contradiction Productivity (gL-T1)**: Contradictions expand possibility spaces
- **Recursive Enhancement (gL-T2)**: Repeated operations increase generativity
- **Non-Explosion (gL4)**: Contradictions don't trivialize reasoning

## Contributing

This is a philosophical prototype. Contributions welcome for:
- Algorithmic enhancements (e.g., ML-based possibility expansion)
- Additional operators or truth lattices
- Integration with symbolic math libraries
- Performance optimizations

## License

Copyright Avery Alexander Rijos. See individual files for licensing.

## Contact

Avery Alexander Rijos - Philosophical and computational explorations of generativity.

---

*From contradiction, anything can follow—but in this system, what follows is possibility.*
