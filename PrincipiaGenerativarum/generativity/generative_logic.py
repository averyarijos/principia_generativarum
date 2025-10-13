"""
Generative Logic System - Enterprise Implementation
A production-ready implementation of Generative Logic that metabolizes
contradictions into enhanced possibilities.

Based on: Principia Generativarum - Transcendental Induction Logics
Author: Avery Alexander Rijos
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from enum import Enum, auto
from abc import ABC, abstractmethod
import logging
from datetime import datetime
from collections import defaultdict
import hashlib
import json

# ========================================================================
# DOMAIN LAYER - Core Generative Logic Entities
# ========================================================================

class GenerativeTruthValue(Enum):
    """
    Multi-dimensional truth value system representing increasing levels
    of generative capacity rather than binary true/false.
    
    Based on information-theoretic foundations:
    - G₀: 0 bits (hinge state - threshold of possibility)
    - G₁: 1 bit (basic generative truth)
    - G₂: 2 bits (enhanced generative truth)
    - G₃: 3 bits (compound generative truth)
    - G∞: ∞ bits (transcendent generative state)
    """
    G_HINGE = 0      # Threshold of possibility - unprocessed contradictions
    G_BASIC = 1      # Basic generative actualization
    G_ENHANCED = 2   # Enhanced generative actualization
    G_COMPOUND = 3   # Compound generative actualization
    G_TRANSCENDENT = 4  # Transcendent generative state
    
    def info_content(self) -> int:
        """Returns the information content in bits."""
        return self.value
    
    def enhance(self, delta: int = 1) -> GenerativeTruthValue:
        """
        Recursive enhancement operation.
        Theorem gL-T2: Recursive application increases generative capacity.
        """
        new_value = min(self.value + delta, GenerativeTruthValue.G_TRANSCENDENT.value)
        return GenerativeTruthValue(new_value)


@dataclass(frozen=True)
class Proposition:
    """
    Immutable proposition with identity based on content hash.
    Supports classical and generative logical operations.
    """
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self) -> int:
        return hash(self.content)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Proposition):
            return NotImplemented
        return self.content == other.content
    
    def __repr__(self) -> str:
        return f"Prop({self.content})"


@dataclass
class GenerativeState:
    """
    Represents the current generative state with truth value,
    possibility space, and metabolic trace.
    """
    proposition: Proposition
    truth_value: GenerativeTruthValue
    possibility_space: Set[Proposition] = field(default_factory=set)
    metabolic_trace: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def log_transformation(self, operation: str) -> None:
        """Records metabolic operations for audit trail."""
        self.metabolic_trace.append(
            f"{datetime.utcnow().isoformat()}: {operation}"
        )


@dataclass
class Contradiction:
    """
    Represents a detected contradiction with its constituents.
    Axiom gL1: Contradictions route through the generative zero operator.
    """
    proposition_a: Proposition
    proposition_b: Proposition
    context: Dict[str, Any] = field(default_factory=dict)
    detected_at: datetime = field(default_factory=datetime.utcnow)
    
    def __repr__(self) -> str:
        return f"⊥({self.proposition_a} ∧ ¬{self.proposition_b})"


@dataclass
class MetabolicScar:
    """
    Represents the residue of a metabolized contradiction.
    The system remembers transformations through scarred memory.
    """
    original_contradiction: Contradiction
    generated_possibilities: Set[Proposition]
    enhancement_delta: int
    metabolized_at: datetime
    scar_id: str = field(default_factory=lambda: hashlib.sha256(
        str(datetime.utcnow().timestamp()).encode()
    ).hexdigest()[:12])


# ========================================================================
# LOGICAL OPERATORS - Core Generative Operations
# ========================================================================

class LogicalOperator(ABC):
    """Abstract base for all logical operators."""
    
    @abstractmethod
    def apply(self, *propositions: Proposition) -> GenerativeState:
        """Apply the operator to one or more propositions."""
        pass


class GenerativeZeroOperator(LogicalOperator):
    """
    The ⊙₀ operator - transforms contradictions into enhanced possibilities.
    Axiom gL2: Metabolic Transformation
    ⊙₀(φ ∧ ¬φ) → Ψ where Ψ expands the possibility space
    """
    
    def __init__(self, expansion_strategy: Callable[[Contradiction], Set[Proposition]]):
        self.expansion_strategy = expansion_strategy
        self.logger = logging.getLogger(__name__)
    
    def apply(self, contradiction: Contradiction) -> GenerativeState:
        """
        Reroutes contradiction through the hinge state,
        transforming impossibility into expanded possibilities.
        """
        self.logger.info(f"Metabolizing contradiction: {contradiction}")
        
        # Generate expanded possibility space
        new_possibilities = self.expansion_strategy(contradiction)
        
        # Create generative state at hinge
        state = GenerativeState(
            proposition=contradiction.proposition_a,  # Primary proposition
            truth_value=GenerativeTruthValue.G_HINGE,
            possibility_space=new_possibilities
        )
        
        state.log_transformation(
            f"⊙₀: Contradiction metabolized → {len(new_possibilities)} new possibilities"
        )
        
        return state


class GenerativeNegationOperator(LogicalOperator):
    """
    The ∇_g operator - transforms propositions into expanded possibility space.
    Unlike classical negation which simply denies, generative negation reveals
    unrealized potential.
    
    Axiom gL3: Recursive Enhancement
    ∇_g(∇_g(φ)) = ∇_g^(2+δ)(φ) where δ > 0
    """
    
    def __init__(self, enhancement_delta: int = 1):
        self.enhancement_delta = enhancement_delta
        self.logger = logging.getLogger(__name__)
    
    def apply(self, proposition: Proposition) -> GenerativeState:
        """
        Generates enhanced possibilities beyond classical negation.
        """
        self.logger.debug(f"Applying generative negation to: {proposition}")
        
        # Generate dialectical possibilities
        base_negation = Proposition(f"¬({proposition.content})")
        synthesis = Proposition(
            f"synthesis({proposition.content}, {base_negation.content})"
        )
        
        possibility_space = {base_negation, synthesis}
        
        state = GenerativeState(
            proposition=proposition,
            truth_value=GenerativeTruthValue.G_BASIC.enhance(self.enhancement_delta),
            possibility_space=possibility_space
        )
        
        state.log_transformation(
            f"∇_g: Generated {len(possibility_space)} dialectical possibilities"
        )
        
        return state


class MetabolicCompositionOperator(LogicalOperator):
    """
    The ⊗ operator - metabolic synthesis of propositions.
    Creates synthesis preserving generative essence of both components
    while enabling emergent properties.
    """
    
    def apply(self, prop_a: Proposition, prop_b: Proposition) -> GenerativeState:
        """
        Metabolically composes two propositions into enhanced synthesis.
        """
        synthesis = Proposition(
            f"({prop_a.content} ⊗ {prop_b.content})",
            metadata={
                "constituents": [prop_a, prop_b],
                "operation": "metabolic_synthesis"
            }
        )
        
        # Inherit and expand possibility spaces
        possibility_space = {synthesis, prop_a, prop_b}
        
        state = GenerativeState(
            proposition=synthesis,
            truth_value=GenerativeTruthValue.G_ENHANCED,
            possibility_space=possibility_space
        )
        
        state.log_transformation(
            f"⊗: Metabolic synthesis of {prop_a} and {prop_b}"
        )
        
        return state


# ========================================================================
# APPLICATION LAYER - Generative Logic Engine
# ========================================================================

class GenerativeLogicEngine:
    """
    Production-ready generative logic system that metabolizes contradictions
    into enhanced possibilities.
    
    Implements:
    - Contradiction detection and metabolism (Axiom gL1, gL2)
    - Recursive enhancement (Axiom gL3)
    - Non-explosion principle (Axiom gL4)
    - Substrate invariance (Axiom gL5)
    """
    
    def __init__(
        self,
        expansion_strategy: Optional[Callable[[Contradiction], Set[Proposition]]] = None,
        logger: Optional[logging.Logger] = None
    ):
        self.logger = logger or logging.getLogger(__name__)
        
        # Operators
        self.zero_operator = GenerativeZeroOperator(
            expansion_strategy or self._default_expansion_strategy
        )
        self.negation_operator = GenerativeNegationOperator()
        self.composition_operator = MetabolicCompositionOperator()
        
        # Knowledge base
        self.propositions: Dict[str, GenerativeState] = {}
        self.contradictions: List[Contradiction] = []
        self.metabolic_archive: List[MetabolicScar] = []
        
        # Metrics
        self.metrics = {
            "contradictions_detected": 0,
            "contradictions_metabolized": 0,
            "enhancements_applied": 0,
            "possibility_expansions": 0
        }
        
        self.logger.info("Generative Logic Engine initialized")
    
    def assert_proposition(
        self,
        proposition: Proposition,
        initial_truth: GenerativeTruthValue = GenerativeTruthValue.G_BASIC
    ) -> GenerativeState:
        """
        Asserts a proposition into the knowledge base.
        Automatically detects and metabolizes contradictions.
        """
        self.logger.debug(f"Asserting: {proposition}")
        
        # Check for contradictions
        contradiction = self._detect_contradiction(proposition)
        
        if contradiction:
            # Axiom gL1: Route through generative zero
            self.metrics["contradictions_detected"] += 1
            return self.metabolize_contradiction(contradiction)
        
        # No contradiction - add to knowledge base
        state = GenerativeState(
            proposition=proposition,
            truth_value=initial_truth
        )
        self.propositions[proposition.content] = state
        
        return state
    
    def metabolize_contradiction(self, contradiction: Contradiction) -> GenerativeState:
        """
        Axiom gL2: Metabolic Transformation
        Transforms contradiction through ⊙₀ operator into enhanced possibilities.
        
        Theorem gL-T1: Contradiction Productivity
        Every contradiction generates enhanced generativity.
        """
        self.logger.info(f"Metabolizing: {contradiction}")
        self.contradictions.append(contradiction)
        
        # Apply generative zero operator
        state = self.zero_operator.apply(contradiction)
        
        # Enhance truth value (anti-fragile property)
        state.truth_value = state.truth_value.enhance(delta=2)
        
        # Create metabolic scar for memory
        scar = MetabolicScar(
            original_contradiction=contradiction,
            generated_possibilities=state.possibility_space,
            enhancement_delta=2,
            metabolized_at=datetime.utcnow()
        )
        self.metabolic_archive.append(scar)
        
        # Update metrics
        self.metrics["contradictions_metabolized"] += 1
        self.metrics["possibility_expansions"] += len(state.possibility_space)
        
        # Update knowledge base with enhanced state
        self.propositions[state.proposition.content] = state
        
        self.logger.info(
            f"Contradiction metabolized → {len(state.possibility_space)} new possibilities"
        )
        
        return state
    
    def apply_generative_negation(self, proposition: Proposition) -> GenerativeState:
        """
        Applies ∇_g operator to generate dialectical possibilities.
        
        Axiom gL3: Recursive Enhancement
        Each application increases generative capacity.
        """
        state = self.negation_operator.apply(proposition)
        
        # Update knowledge base
        self.propositions[proposition.content] = state
        self.metrics["enhancements_applied"] += 1
        
        return state
    
    def metabolic_synthesis(
        self,
        prop_a: Proposition,
        prop_b: Proposition
    ) -> GenerativeState:
        """
        Composes propositions through ⊗ operator.
        Creates synthesis that preserves generative essence.
        """
        state = self.composition_operator.apply(prop_a, prop_b)
        
        # Update knowledge base
        self.propositions[state.proposition.content] = state
        
        return state
    
    def recursive_enhance(
        self,
        proposition: Proposition,
        iterations: int = 1
    ) -> GenerativeState:
        """
        Theorem gL-T2: Recursive Generativity
        Recursive application increases generative capacity.
        
        G^n(φ) = G^(n+1+δ)(φ) where δ > 0
        """
        state = self.propositions.get(
            proposition.content,
            GenerativeState(
                proposition=proposition,
                truth_value=GenerativeTruthValue.G_BASIC
            )
        )
        
        for i in range(iterations):
            state.truth_value = state.truth_value.enhance(delta=1)
            state.log_transformation(f"Recursive enhancement iteration {i+1}")
            self.metrics["enhancements_applied"] += 1
        
        self.propositions[proposition.content] = state
        
        return state
    
    def query_possibility_space(
        self,
        proposition: Proposition
    ) -> Optional[Set[Proposition]]:
        """
        Retrieves the possibility space generated by a proposition.
        """
        state = self.propositions.get(proposition.content)
        return state.possibility_space if state else None
    
    def get_metabolic_history(self) -> List[MetabolicScar]:
        """
        Returns complete metabolic archive (scarred memory).
        The system remembers all transformations.
        """
        return self.metabolic_archive.copy()
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Anti-Fragile Validity: System health increases with contradiction rate.
        
        SystemHealth ∝ ContradictionMetabolizationRate
        """
        total_contradictions = self.metrics["contradictions_detected"]
        metabolized = self.metrics["contradictions_metabolized"]
        
        metabolization_rate = (
            metabolized / total_contradictions if total_contradictions > 0 else 1.0
        )
        
        # Health increases with successful metabolism
        health_score = min(metabolization_rate * 100, 100)
        
        return {
            "health_score": health_score,
            "vitality": "robust" if health_score > 80 else "moderate" if health_score > 50 else "fragile",
            "metrics": self.metrics.copy(),
            "knowledge_base_size": len(self.propositions),
            "metabolic_scars": len(self.metabolic_archive)
        }
    
    def export_state(self) -> Dict[str, Any]:
        """
        Exports complete system state for persistence or inspection.
        """
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "propositions": {
                k: {
                    "content": v.proposition.content,
                    "truth_value": v.truth_value.name,
                    "possibility_space": [p.content for p in v.possibility_space],
                    "metabolic_trace": v.metabolic_trace
                }
                for k, v in self.propositions.items()
            },
            "metrics": self.metrics,
            "health": self.get_system_health()
        }
    
    # Private methods
    
    def _detect_contradiction(
        self,
        proposition: Proposition
    ) -> Optional[Contradiction]:
        """
        Detects classical contradictions (φ ∧ ¬φ).
        Axiom gL4: Non-Explosion - contradictions don't imply everything.
        """
        # Simple negation detection
        for existing_prop in self.propositions.keys():
            # Check if this is a negation of an existing proposition
            if (proposition.content == f"¬({existing_prop})" or
                existing_prop == f"¬({proposition.content})"):
                
                return Contradiction(
                    proposition_a=proposition,
                    proposition_b=Proposition(existing_prop)
                )
        
        return None
    
    @staticmethod
    def _default_expansion_strategy(contradiction: Contradiction) -> Set[Proposition]:
        """
        Default strategy for expanding possibility space from contradictions.
        Generates dialectical synthesis and alternative framings.
        """
        prop_a = contradiction.proposition_a
        prop_b = contradiction.proposition_b
        
        # Generate synthetic possibilities
        return {
            Proposition(f"synthesis({prop_a.content}, {prop_b.content})"),
            Proposition(f"context_A_holds({prop_a.content})"),
            Proposition(f"context_B_holds({prop_b.content})"),
            Proposition(f"temporal_sequence({prop_a.content}, {prop_b.content})"),
            Proposition(f"higher_order_resolution({prop_a.content}, {prop_b.content})")
        }


# ========================================================================
# INFRASTRUCTURE LAYER - Observability & Deployment
# ========================================================================

class GenerativeLogicMetricsCollector:
    """
    Prometheus-compatible metrics collector for observability.
    """
    
    def __init__(self, engine: GenerativeLogicEngine):
        self.engine = engine
    
    def collect_metrics(self) -> Dict[str, Union[int, float]]:
        """Returns metrics in Prometheus format."""
        health = self.engine.get_system_health()
        
        return {
            "generative_logic_health_score": health["health_score"],
            "generative_logic_contradictions_detected_total": self.engine.metrics["contradictions_detected"],
            "generative_logic_contradictions_metabolized_total": self.engine.metrics["contradictions_metabolized"],
            "generative_logic_enhancements_applied_total": self.engine.metrics["enhancements_applied"],
            "generative_logic_possibility_expansions_total": self.engine.metrics["possibility_expansions"],
            "generative_logic_knowledge_base_size": len(self.engine.propositions),
            "generative_logic_metabolic_scars_total": len(self.engine.metabolic_archive)
        }


# ========================================================================
# USAGE EXAMPLES
# ========================================================================

def main():
    """Demonstrates generative logic capabilities."""
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize engine
    engine = GenerativeLogicEngine()
    
    print("=" * 80)
    print("GENERATIVE LOGIC ENGINE - Production Demo")
    print("=" * 80)
    
    # Example 1: Basic assertion
    print("\n[1] Asserting proposition...")
    prop1 = Proposition("The system is consistent")
    state1 = engine.assert_proposition(prop1)
    print(f"   Truth Value: {state1.truth_value.name}")
    
    # Example 2: Contradiction detection and metabolism
    print("\n[2] Introducing contradiction...")
    prop2 = Proposition("¬(The system is consistent)")
    state2 = engine.assert_proposition(prop2)
    print(f"   Contradiction detected and metabolized!")
    print(f"   New Truth Value: {state2.truth_value.name}")
    print(f"   Possibility Space: {len(state2.possibility_space)} new possibilities")
    for p in state2.possibility_space:
        print(f"      • {p.content}")
    
    # Example 3: Generative negation
    print("\n[3] Applying generative negation...")
    prop3 = Proposition("Classical logic is complete")
    state3 = engine.apply_generative_negation(prop3)
    print(f"   Generated possibilities:")
    for p in state3.possibility_space:
        print(f"      • {p.content}")
    
    # Example 4: Recursive enhancement
    print("\n[4] Recursive enhancement...")
    prop4 = Proposition("Intelligence emerges from complexity")
    state4 = engine.recursive_enhance(prop4, iterations=3)
    print(f"   Enhanced Truth Value: {state4.truth_value.name}")
    print(f"   Metabolic Trace:")
    for trace in state4.metabolic_trace:
        print(f"      {trace}")
    
    # Example 5: System health (anti-fragile)
    print("\n[5] System Health (Anti-Fragile Validity)...")
    health = engine.get_system_health()
    print(f"   Health Score: {health['health_score']:.2f}/100")
    print(f"   Vitality: {health['vitality']}")
    print(f"   Metrics: {json.dumps(health['metrics'], indent=6)}")
    
    # Example 6: Export state
    print("\n[6] Exporting system state...")
    state_export = engine.export_state()
    print(f"   Total Propositions: {len(state_export['propositions'])}")
    print(f"   Metabolic Scars: {state_export['health']['metabolic_scars']}")
    
    print("\n" + "=" * 80)
    print("Demo complete. Contradiction → Enhancement achieved.")
    print("=" * 80)


if __name__ == "__main__":
    main()
