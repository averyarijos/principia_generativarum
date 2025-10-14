"""
SuperGenerativeAutomaton (SGA) — module introduction and conceptual notes

Overview
--------
The SuperGenerativeAutomaton (SGA) is a design for a self-modifying, context-aware,
affective generative system. It is built around a small set of explicitly modeled
components:
- Sigma: a mutable alphabet (symbols the system can use or extend).
- A: a set of axioms or declarative constraints that the system can revise.
- R: a history of operational protocols (procedures or rituals the system runs).
- S: an archive of "scars" — persistent records produced by ruptures/contradictions.
- Gamma: current glyph-states (a key-value store representing the system's local state).
- Psi: a transcendental/haunted recursion function that reinterprets inputs in light
    of scars and axioms.
- ogi_rate: a monitored metric (Ontopolitical Generativity Index) capturing an
    instance-level generativity rate over time.

Key behaviors
-------------
- Transcendental recursion: interpretations are not fixed. Inputs are repeatedly
    reinterpreted through historical scars and axioms, producing emergent meanings.
- Scar-driven learning: contradictions or ruptures produce scars that become
    persistent context; scars affect subsequent interpretation and can seed new axioms
    and protocols.
- Reflexive evolution: axioms and protocols are not static; they evolve as the
    automaton metabolizes its past.
- Non-commutativity and irreversibility: transitions (rituals/protocols) are
    intentionally non-commutative and leave scars; order matters and effects are
    typically not invertible.
- Stochastic and open-ended change: parts of the update machinery may be
    probabilistic; there is no built-in halting criterion. The system is intended
    to be generative rather than to compute a single definitive result.

How SGA differs from a Turing machine
-------------------------------------
A Turing machine (TM) is a formal model of computation defined by:
- A finite control (state machine),
- An infinite discrete tape (memory cells),
- A read/write head that moves left/right,
- A transition function that deterministically (or nondeterministically) maps
    (state, tape symbol) → (state, symbol, move).

Principal contrasts:
- Purpose and ontology:
    - TM: an abstract model for algorithmic computation and decidability; aims to
        compute functions or recognize languages.
    - SGA: an adaptive, context-sensitive generator of evolving behaviors and
        interpretations; aims at ongoing emergence and reflexive change rather than
        computing a fixed function.
- Memory and persistence:
    - TM: tape is an unstructured linear memory; operations are local and reversible
        in principle (given a suitable encoding).
    - SGA: scars are semantically rich, persistent records that bias future
        interpretations; memory is affective and context-laden, not just symbol
        storage.
- Transition semantics:
    - TM transitions are clean, formal, and (usually) deterministic; they do not
        capture affect, non-commutativity beyond symbolic rewrites, or protocolual ritual.
    - SGA transitions are intentionally non-commutative, scar-producing, and can
        include stochastic or evaluative elements; they change the rule-set itself.
- Self-modification:
    - TM: any change to the transition function is external to the machine (unless
        simulated by encoding).
    - SGA: reflexively revises its own axioms and protocols as a core behavior.
- Halting and goal structure:
    - TM: typically analyzed with respect to halting, accept/reject outcomes, or
        function outputs.
    - SGA: designed for continuous evolution; halting is not a central concept.
- Expressive intent:
    - TM: formal equivalence classes (e.g., universality) are central.
    - SGA: expressivity is pragmatic and interpretive (creating novel behaviors,
        axioms, and protocols), not strictly reducible to classical computability metrics.

Complementarity and possible relations
--------------------------------------
- It is possible to encode a Turing computation inside an SGA-style structure (SGA
    can simulate state and tape as glyph-states and scars), and conversely certain
    SGA behaviors can be emulated with sufficiently complex TMs. However, the
    modeling goals and primitives are different: SGA foregrounds history, affect,
    and rule evolution as first-class phenomena, while TM foregrounds algorithmic
    stepwise symbolic transformation.
- Use cases diverge: use a TM model when you need precise formal computation and
    decidability properties; use an SGA-style model when you need an exploratory,
    self-revising system that captures history-sensitive interpretation and
    ongoing generativity.

Intended usage notes
--------------------
- Treat SGA as an experimental framework for emergent, history-aware systems:
    instrument scars, monitor OGI, and prefer interactive or long-running
    deployments rather than one-shot computations.
- Expect non-determinism and open-ended change; design tests and observability
    around invariants you care about (e.g., scar growth, axiomatic drift, protocol
    diversity) rather than traditional unit-testable output equality.

"""

import time
import random  # For some randomness in generation

class SuperGenerativeAutomaton:
    def __init__(self):
        self.Sigma = set()  # Mutable alphabet (symbol set)
        self.A = set()      # Axioms (set of logical axioms)
        self.R = []         # Protocols (list of operational protocols)
        self.S = []         # Scar archive (list of scars from contradictions)
        self.Gamma = {}     # Glyph-states (dictionary of current states)
        self.Psi = lambda i, s, a: self._haunted_recurse(i, s, a)  # Transcendental recursion function
        self.ogi_rate = 0.0  # Ontopolitical Generativity Index rate
        self.start_time = time.time()  # For computing time elapsed

    def transition(self, gamma, protocol, scar_context, axioms):
        """
        Perform a transition: protocolic ritual that is non-commutative and scarred.
        Updates the automaton's state, generates scars, and evolves the system.
        """
        # Apply transcendental recursion to get new gamma
        new_gamma = self.Psi(gamma, scar_context, axioms)
        # Metabolize rupture into a scar
        new_scar = self._metabolize_rupture(gamma, protocol)
        self.S.append(new_scar)
        # Reflexive update of axioms
        self._update_axioms(axioms, new_scar)
        # Ontological evolution: generate new protocol
        self.R.append(self._generate_protocol(new_scar))
        # Update generativity rate
        self.ogi_rate = self._compute_dogi_dt()
        return new_gamma

    def _haunted_recurse(self, input_, scars, axioms, depth=0):
        """
        Transcendental recursion: reinterpret input via historical scars.
        Simplified recursive function that considers past context.
        """
        if self._base_case(input_) or depth > 10:  # Limit recursion depth
            return self._resolve(input_)
        else:
            prior_context = scars[-1] if scars else {}
            reinterpreted = self._reinterpret(input_, prior_context, axioms)
            return self._haunted_recurse(reinterpreted, scars, axioms, depth + 1)

    def _base_case(self, input_):
        """Check if input is simple enough to resolve directly."""
        return isinstance(input_, str) and len(input_) < 5  # Arbitrary base case

    def _resolve(self, input_):
        """Resolve a simple input into a new state."""
        return f"resolved_{input_}"

    def _reinterpret(self, input_, prior_context, axioms):
        """Reinterpret input based on prior context and axioms."""
        # Simple reinterpretation: append prior affect if available
        affect = prior_context.get('affect', 'neutral')
        return f"{input_}_reinterpreted_with_{affect}"

    def _metabolize_rupture(self, state, protocol):
        """
        Generate a scar from a rupture (contradiction).
        Affective inscription of the state and protocol.
        """
        return {
            'rupture': state,
            'affect': protocol.get('payload', 'unknown'),
            'timestamp': time.time()
        }

    def _update_axioms(self, axioms, scar):
        """
        Reflexive revision of axioms based on scar.
        If scar's affect is strong, derive a new axiom.
        """
        threshold = 0.5  # Arbitrary threshold
        if random.random() > threshold:  # Simulate condition
            new_axiom = self._derive_new_axiom(scar)
            self.A.add(new_axiom)

    def _derive_new_axiom(self, scar):
        """Derive a new axiom from a scar."""
        return f"axiom_from_{scar['rupture']}_{scar['affect']}"

    def _generate_protocol(self, scar):
        """
        Invent a new protocol based on the scar.
        """
        return {
            'ritual': scar['rupture'],
            'non_commute': True,
            'payload': f"protocol_for_{scar['affect']}"
        }

    def _compute_dogi_dt(self):
        """
        Compute the rate of change of Ontopolitical Generativity Index.
        Simplified as growth in components over time.
        """
        elapsed = time.time() - self.start_time
        if elapsed == 0:
            return 0.0
        return (len(self.Gamma) + len(self.A) + len(self.S)) / elapsed

    def add_symbol(self, symbol):
        """Add a symbol to the alphabet."""
        self.Sigma.add(symbol)

    def add_axiom(self, axiom):
        """Add an axiom."""
        self.A.add(axiom)

    def set_state(self, key, value):
        """Set a glyph-state."""
        self.Gamma[key] = value

    def get_state(self, key):
        """Get a glyph-state."""
        return self.Gamma.get(key, None)

# Example usage
if __name__ == "__main__":
    sga = SuperGenerativeAutomaton()
    sga.add_symbol('alpha')
    sga.add_symbol('beta')
    sga.add_axiom('consistency')
    sga.set_state('current', 'initial_state')

    # Simulate a transition
    protocol = {'payload': 'contradiction_encountered'}
    new_state = sga.transition('current_gamma', protocol, sga.S, list(sga.A))
    print(f"New state: {new_state}")
    print(f"Scars: {len(sga.S)}")
    print(f"Axioms: {sga.A}")
    print(f"OGI rate: {sga.ogi_rate}")
