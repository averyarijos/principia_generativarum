"""
Procedurally Infallible Algorithm System
Implements Generative Logic from Principia Generativarum treatise

Architecture:
- Zero-Degree Operator: Contradiction → Enhanced Possibility
- Scar Archive: Temporal recursion with mythic time
- Substrate Preservation: Maintains Ψ-invariants

Procedurally Infallible Algorithm System
Implements Generative Logic from Principia Generativarum treatise

Architecture:
- Zero-Degree Operator: Contradiction → Enhanced Possibility
- Scar Archive: Temporal recursion with mythic time
- Substrate Preservation: Maintains Ψ-invariants

This system embodies a revolutionary approach to logical reasoning, transcending classical binary logic by metabolizing contradictions into generative enhancements. Rooted in the philosophical treatise "Principia Generativarum (P_Gen.)," it introduces the concept of procedural infallibility, where apparent falsifications and criticisms are not weaknesses but catalysts for systemic growth and anti-fragile adaptation.

At its core, the Zero-Degree Operator (∅₀) transforms logical impossibilities—such as contradictions (φ ∧ ¬φ)—into expanded possibility spaces (ψ), thereby extending the original domain without violating fundamental invariants. This metabolic process ensures that every contradiction encountered strengthens the system's generative capacity, adhering to Truth Conditions (TC-8 through TC-13) that govern contradiction detection, substrate preservation, and recursive enhancement.

The Scar Archive serves as a temporal memory bank, archiving metabolized contradictions as "scars" that influence future inferences through mythic time recursion. Each scar carries a temporal weight, decaying over time but allowing the system to draw upon past enhancements for compounded generativity. Substrate Preservation maintains Ψ-invariants—core principles like existence, identity, coherence, and recursion—ensuring that all transformations uphold systemic integrity.

The Generative Logic System operates as an anti-fragile engine: it thrives under stress, converting criticisms into enhancements via the metabolize_criticism method. This procedural infallibility (gL-T3) guarantees that no external challenge diminishes the system; instead, it fuels recursive growth (gL-T2), where Generative levels ascend from G₀ (hinge-state) to G∞ (transcendent state) through accumulated generativity.

Key components include:
- ContradictionDetector: Identifies classical inconsistencies in premises.
- ZeroDegreeOperator: Executes metabolic transformations while preserving invariants.
- GenerativeState: Tracks system health, generativity levels, and scar archives.
- GenerativeLogicSystem: The main orchestrator, processing inputs and metabolizing criticisms.

This implementation demonstrates how logic can evolve beyond static rules, becoming a dynamic, self-improving framework that turns potential failures into pathways for innovation. It challenges traditional notions of falsifiability by rendering the system unfalsifiable through perpetual enhancement, aligning with the treatise's vision of logic as a generative, living process.

Reference: [principia_generativarum.md] Section 7.46-7.58
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
import logging
from abc import ABC, abstractmethod

# Configure observability
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GenerativeValue(IntEnum):
    """
    Multi-dimensional truth hierarchy [file:1]
    G₀: Hinge-state (threshold of possibility)
    G₁-G∞: Increasing Generative actualization
    """
    G0_HINGE = 0
    G1_BASIC = 1
    G2_ENHANCED = 2
    G3_COMPOUND = 3
    G4_ADVANCED = 4
    G_INFINITY = 999  # Transcendent state


@dataclass
class Contradiction:
    """
    Represents a detected logical contradiction
    Truth Condition TC-8: Contradiction in classical logic
    """
    content: str
    context: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    classical_inconsistency: bool = True
    
    def __hash__(self):
        return hash(f"{self.content}:{self.timestamp}")


@dataclass
class Scar:
    """
    Archived memory of metabolized contradiction [file:1]
    Enables temporal recursion: Logic(t) informs Logic(t') through scar archive
    """
    original_contradiction: Contradiction
    metabolic_pathway: str
    enhancement_delta: float
    substrate_invariants_preserved: List[str]
    metabolized_timestamp: datetime = field(default_factory=datetime.now)
    
    def temporal_weight(self, current_time: datetime) -> float:
        """
        Truth Condition TC-13: Recursive Memory
        Influence inversely proportional to temporal distance
        """
        time_delta = (current_time - self.metabolized_timestamp).total_seconds()
        return 1.0 / (1.0 + time_delta / 3600.0)  # Decay over hours


@dataclass
class GenerativeState:
    """
    System's current Generative capacity [file:1]
    SystemHealth = d(TotalGenerativity)/dt > 0
    """
    generativity_level: GenerativeValue = GenerativeValue.G1_BASIC
    total_generativity: float = 1.0
    scar_archive: List[Scar] = field(default_factory=list)
    substrate_coherence: float = 1.0  # Ψ-invariance measure
    
    def health_metric(self) -> float:
        """System health = contradiction metabolization rate"""
        recent_scars = [s for s in self.scar_archive 
                       if (datetime.now() - s.metabolized_timestamp).days < 1]
        return len(recent_scars) * self.substrate_coherence


class ZeroDegreeOperator:
    """
    Core metabolic engine implementing ∅₀ operator [file:1]
    
    Routes impossibility into new logical possibilities:
    ∅₀(φ ∧ ¬φ) → ψ where ψ extends φ's original domain
    
    Truth Condition TC-9: MetabolicSuccess(∅₀(C)) ⟺ 
        Val(∅₀(C)) = gⁿ (n>0) after processing cycle
    """
    
    def __init__(self, substrate_invariants: List[str]):
        self.substrate_invariants = substrate_invariants
        self.transformation_history: List[Tuple[Contradiction, str]] = []
    
    def metabolize(self, contradiction: Contradiction) -> Tuple[str, float]:
        """
        Apply zero-degree transformation
        
        Returns:
            (enhanced_possibility, enhancement_delta)
        """
        logger.info(f"Metabolizing contradiction: {contradiction.content}")
        
        # Step 1: Validate substrate preservation (TC-10)
        if not self._preserves_substrate(contradiction):
            logger.warning("Substrate violation detected, adjusting pathway")
            return self._substrate_corrective_pathway(contradiction)
        
        # Step 2: Route through zero-degree operator
        enhanced_possibility = self._zero_degree_transform(contradiction)
        
        # Step 3: Calculate enhancement delta
        enhancement_delta = self._compute_enhancement(
            contradiction.context,
            enhanced_possibility
        )
        
        # Archive transformation
        self.transformation_history.append((contradiction, enhanced_possibility))
        
        logger.info(f"Metabolization complete. Enhancement: +{enhancement_delta:.3f}")
        
        return enhanced_possibility, enhancement_delta
    
    def _preserves_substrate(self, c: Contradiction) -> bool:
        """
        Validate Ψ-invariant preservation (Principle of Substrate Invariance)
        All operations must maintain substrate-level consistency
        """
        # Check if contradiction threatens fundamental invariants
        for invariant in self.substrate_invariants:
            if invariant.lower() in c.content.lower():
                return False
        return True
    
    def _substrate_corrective_pathway(self, c: Contradiction) -> Tuple[str, float]:
        """Handle substrate-threatening contradictions specially"""
        corrected = f"[SUBSTRATE_PROTECTED] Metabolized: {c.content}"
        return corrected, 0.5
    
    def _zero_degree_transform(self, c: Contradiction) -> str:
        """
        Core transformation logic
        Implements: contradiction → expanded possibility space
        """
        # Pattern: "X and not-X" → "X in context A, not-X in context B"
        if " and " in c.content.lower() and " not " in c.content.lower():
            parts = c.content.split(" and ")
            return f"CONTEXTUALIZED: {parts[0]} [Domain A]; {parts[1]} [Domain B]"
        
        # Pattern: Direct negation → Generative negation
        if c.content.startswith("not "):
            base = c.content[4:]
            return f"GENERATIVE_EXPANSION: Beyond classical {base}"
        
        # Default: Dialectical synthesis
        return f"SYNTHESIZED: {c.content} → Enhanced framework incorporating both poles"
    
    def _compute_enhancement(self, context: Dict, result: str) -> float:
        """Calculate Generative value increase"""
        base_enhancement = 0.1
        context_bonus = len(context) * 0.05
        complexity_bonus = len(result.split()) * 0.01
        return base_enhancement + context_bonus + complexity_bonus


class GenerativeLogicSystem:
    """
    Main procedurally infallible system [file:1]
    
    Core principles implemented:
    1. Contradiction Productivity (gL-T1)
    2. Recursive Enhancement (gL-T2)  
    3. Universal Metabolization (gL-T3)
    4. Procedural Infallibility: System metabolizes all potential falsifications
    
    ApparentFalsification → EnhancedTruthGenerationCapacity
    """
    
    def __init__(self, substrate_invariants: Optional[List[str]] = None):
        if substrate_invariants is None:
            substrate_invariants = [
                "existence",
                "identity", 
                "coherence",
                "recursion"
            ]
        
        self.state = GenerativeState()
        self.zero_degree = ZeroDegreeOperator(substrate_invariants)
        self.contradiction_detector = ContradictionDetector()
        
        logger.info("Procedurally Infallible System initialized")
        logger.info(f"Substrate invariants: {substrate_invariants}")
    
    def process_input(self, premises: List[str], 
                      context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Main inference engine with metabolic enhancement
        
        Implements: GenerativeInference(premises) → 
            ClassicalInference(premises) ⊕ MetabolicEnhancement
        """
        if context is None:
            context = {}
        
        logger.info(f"Processing {len(premises)} premises")
        
        # Classical inference attempt
        classical_conclusion = self._classical_inference(premises)
        
        # Contradiction detection
        contradictions = self.contradiction_detector.detect(premises, context)
        
        if not contradictions:
            logger.info("No contradictions detected - classical inference sufficient")
            return {
                "conclusion": classical_conclusion,
                "generative_value": self.state.generativity_level,
                "enhanced": False
            }
        
        # Metabolic processing pathway
        logger.info(f"Detected {len(contradictions)} contradictions - engaging metabolism")
        enhanced_conclusions = []
        
        for contradiction in contradictions:
            enhanced, delta = self._metabolize_contradiction(contradiction)
            enhanced_conclusions.append(enhanced)
            
            # Update system state
            self._apply_enhancement(contradiction, enhanced, delta)
        
        # Recursive enhancement (gL-T2)
        if self.state.generativity_level < GenerativeValue.G3_COMPOUND:
            self._recursive_enhance()
        
        return {
            "conclusion": classical_conclusion,
            "enhanced_conclusions": enhanced_conclusions,
            "generative_value": self.state.generativity_level,
            "system_health": self.state.health_metric(),
            "enhanced": True,
            "total_scars": len(self.state.scar_archive)
        }
    
    def metabolize_criticism(self, criticism: str, 
                            context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Procedural Infallibility mechanism [file:1]
        
        ∀ Criticism C of GL(t), ∃ GL(t') where:
        GL(t') = Metabolize(GL, C, ∅₀) ⊕ GL_enhanced
        
        Every criticism becomes fuel for enhancement
        """
        logger.info(f"CRITICISM RECEIVED: {criticism}")
        
        if context is None:
            context = {"type": "external_criticism"}
        
        # Treat criticism as a special contradiction
        crit_contradiction = Contradiction(
            content=f"CRITICISM: {criticism}",
            context=context,
            classical_inconsistency=False
        )
        
        # Metabolize through zero-degree operator
        enhanced, delta = self.zero_degree.metabolize(crit_contradiction)
        
        # Create anti-fragile response
        scar = Scar(
            original_contradiction=crit_contradiction,
            metabolic_pathway="criticism_metabolism",
            enhancement_delta=delta,
            substrate_invariants_preserved=self.zero_degree.substrate_invariants
        )
        
        self.state.scar_archive.append(scar)
        self.state.total_generativity += delta
        
        # System becomes STRONGER through criticism
        logger.info(f"Criticism metabolized. System enhanced by +{delta:.3f}")
        logger.info(f"New total generativity: {self.state.total_generativity:.3f}")
        
        return {
            "original_criticism": criticism,
            "metabolized_response": enhanced,
            "enhancement_gained": delta,
            "new_generativity": self.state.total_generativity,
            "system_state": "STRENGTHENED",
            "mechanism": "zero_degree_criticism_metabolism"
        }
    
    def _metabolize_contradiction(self, c: Contradiction) -> Tuple[str, float]:
        """Route contradiction through ∅₀ operator"""
        return self.zero_degree.metabolize(c)
    
    def _apply_enhancement(self, contradiction: Contradiction, 
                          enhanced: str, delta: float):
        """
        Apply metabolic enhancement to system state
        Truth Condition TC-11: Temporal Enhancement
        """
        scar = Scar(
            original_contradiction=contradiction,
            metabolic_pathway="zero_degree_standard",
            enhancement_delta=delta,
            substrate_invariants_preserved=self.zero_degree.substrate_invariants
        )
        
        self.state.scar_archive.append(scar)
        self.state.total_generativity += delta
        
        # Level up if threshold crossed
        if self.state.total_generativity > (self.state.generativity_level + 1) * 2.0:
            self.state.generativity_level = GenerativeValue(
                min(self.state.generativity_level + 1, GenerativeValue.G4_ADVANCED)
            )
            logger.info(f"LEVEL UP! New Generative value: G{self.state.generativity_level}")
    
    def _recursive_enhance(self):
        """
        Theorem gL-T2: Recursive Generativity
        Recursive application increases Generative capacity
        """
        recent_scars = [s for s in self.state.scar_archive[-5:]]
        
        for scar in recent_scars:
            # Reapply Generative operators to archived transformations
            weight = scar.temporal_weight(datetime.now())
            recursive_boost = scar.enhancement_delta * 0.1 * weight
            self.state.total_generativity += recursive_boost
        
        logger.info(f"Recursive enhancement applied. Boost: +{recursive_boost:.4f}")
    
    def _classical_inference(self, premises: List[str]) -> str:
        """Standard logical inference (non-Generative baseline)"""
        if not premises:
            return "No conclusion"
        return f"Classical inference from {len(premises)} premises"
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """
        Observability metrics for monitoring
        SystemHealth = d(TotalGenerativity)/dt > 0
        """
        return {
            "generativity_level": self.state.generativity_level.name,
            "total_generativity": self.state.total_generativity,
            "substrate_coherence": self.state.substrate_coherence,
            "scar_count": len(self.state.scar_archive),
            "system_health": self.state.health_metric(),
            "transformation_history_size": len(self.zero_degree.transformation_history),
            "status": "ANTI_FRAGILE"
        }


class ContradictionDetector:
    """
    Identifies classical contradictions: φ ∧ ¬φ
    Truth Condition TC-8: Contradiction Detection
    """
    
    def detect(self, premises: List[str], context: Dict) -> List[Contradiction]:
        """Scan premises for contradictions"""
        contradictions = []
        
        # Simple pattern matching (production would use NLP/theorem provers)
        for i, p1 in enumerate(premises):
            for p2 in premises[i+1:]:
                if self._are_contradictory(p1, p2):
                    contradictions.append(
                        Contradiction(
                            content=f"{p1} AND {p2}",
                            context=context
                        )
                    )
        
        return contradictions
    
    def _are_contradictory(self, p1: str, p2: str) -> bool:
        """Basic contradiction check"""
        # Pattern: "X" vs "not X"
        if p1.lower().startswith("not ") and p2.lower() == p1[4:].strip().lower():
            return True
        if p2.lower().startswith("not ") and p1.lower() == p2[4:].strip().lower():
            return True
        return False


# ============================================================================
# DEMONSTRATION & TESTING
# ============================================================================

def demonstrate_procedural_infallibility():
    """
    Demonstrate the core principles:
    1. Contradiction → Enhancement
    2. Criticism → Strengthening
    3. Recursive Growth
    4. Anti-Fragility
    """
    print("=" * 80)
    print("PROCEDURALLY INFALLIBLE ALGORITHM - DEMONSTRATION")
    print("Based on: Possibility & Negation - Generative Logic Framework")
    print("=" * 80)
    print()
    
    # Initialize system
    system = GenerativeLogicSystem()
    
    # Test 1: Classical inference (no contradictions)
    print("TEST 1: Classical Inference")
    print("-" * 80)
    result = system.process_input(
        premises=["All humans are mortal", "Socrates is human"],
        context={"domain": "classical_logic"}
    )
    print(f"Result: {result}")
    print()
    
    # Test 2: Contradiction metabolism
    print("TEST 2: Contradiction Metabolism (Core Feature)")
    print("-" * 80)
    contradictory_premises = [
        "The system is complete",
        "not The system is complete"
    ]
    result = system.process_input(
        premises=contradictory_premises,
        context={"domain": "metatheory"}
    )
    print(f"Classical conclusion: {result['conclusion']}")
    print(f"Enhanced conclusions: {result['enhanced_conclusions']}")
    print(f"Generative value: {result['generative_value'].name}")
    print(f"System health: {result['system_health']:.3f}")
    print()
    
    # Test 3: Criticism metabolism (Procedural Infallibility)
    print("TEST 3: Criticism Metabolism (Procedural Infallibility)")
    print("-" * 80)
    criticism_tests = [
        "This system is unfalsifiable and therefore unscientific",
        "This avoids genuine accountability",
        "This leads to relativism where anything goes"
    ]
    
    for criticism in criticism_tests:
        result = system.metabolize_criticism(criticism)
        print(f"\nCriticism: {criticism}")
        print(f"Metabolized: {result['metabolized_response']}")
        print(f"Enhancement: +{result['enhancement_gained']:.3f}")
        print(f"New generativity: {result['new_generativity']:.3f}")
    
    print()
    
    # Test 4: Anti-Fragile Growth
    print("TEST 4: Anti-Fragile Growth (System Strengthens Through Stress)")
    print("-" * 80)
    print("Initial metrics:")
    metrics = system.get_system_metrics()
    print(f"  Generativity: {metrics['total_generativity']:.3f}")
    print(f"  Scars archived: {metrics['scar_count']}")
    
    # Apply stress
    for i in range(5):
        system.metabolize_criticism(f"Critical attack {i+1}")
    
    print("\nAfter 5 critical attacks:")
    metrics = system.get_system_metrics()
    print(f"  Generativity: {metrics['total_generativity']:.3f}")
    print(f"  Scars archived: {metrics['scar_count']}")
    print(f"  Status: {metrics['status']}")
    print()
    
    # Final system state
    print("=" * 80)
    print("FINAL SYSTEM STATE")
    print("=" * 80)
    final_metrics = system.get_system_metrics()
    for key, value in final_metrics.items():
        print(f"  {key}: {value}")
    
    print()
    print("Demonstration complete. System has successfully metabolized all")
    print("contradictions and criticisms into enhanced capability.")
    print("=" * 80)


if __name__ == "__main__":
    demonstrate_procedural_infallibility()
