"""
SCAR: Super-Generative Automaton Runtime
Enterprise System for Generative Contradiction Metabolism

Based on Principia Generativarum - Possibility & Negation Framework
"""

import os
import uuid
import time
from typing import Dict, List, Callable, Any, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import json
import logging
from abc import ABC, abstractmethod

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION & LOGGING
# ═══════════════════════════════════════════════════════════════

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("scar.core")


# ═══════════════════════════════════════════════════════════════
# DOMAIN MODEL
# ═══════════════════════════════════════════════════════════════

class PermissionStatus(Enum):
    """Permission evaluation for scar metabolism."""
    DENIED = 0
    AUTHORIZED = 1


@dataclass(frozen=True)
class Contradiction:
    """
    Represents a structured absence or rupture in system coherence.
    
    Attributes:
        id: Unique identifier for the contradiction
        description: Human-readable description of the rupture
        context: Contextual metadata about the contradiction
        severity: Numerical severity (higher = more severe)
    """
    id: str
    description: str
    context: Dict[str, Any]
    severity: float = 1.0
    
    def __repr__(self) -> str:
        return f"Contradiction({self.id}, severity={self.severity})"


@dataclass
class MetabolicProtocol:
    """
    Defines how a scar rewrites the system's transition function.
    
    The protocol μ: δ → δ' transforms the transition function when
    authorized by the permission function ψ.
    """
    name: str
    rewrite_fn: Callable[[Any, "Scar"], Any]
    description: str = ""
    
    def apply(self, delta: Any, scar: "Scar") -> Any:
        """
        Apply the metabolic transformation.
        
        Args:
            delta: Current transition function
            scar: The scar being metabolized
            
        Returns:
            Transformed transition function δ'
        """
        logger.info(f"Applying protocol '{self.name}' to scar {scar.id}")
        return self.rewrite_fn(delta, scar)


@dataclass
class Scar:
    """
    Formal scar tuple: σ = (c, τ, μ, ψ)
    
    A scar is a structured record of rupture with metabolic protocol.
    Only scars where ψ(σ) = 1 are authorized for system transformation.
    
    Attributes:
        id: Unique identifier
        contradiction: The rupture or structured absence
        timestamp: Temporal index τ
        protocol: Metabolic rewrite rule μ
        permission: Authorization status ψ(σ) ∈ {0,1}
    """
    id: str
    contradiction: Contradiction
    timestamp: float
    protocol: MetabolicProtocol
    permission: PermissionStatus = PermissionStatus.DENIED
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_authorized(self) -> bool:
        """Check if scar is authorized for metabolism."""
        return self.permission == PermissionStatus.AUTHORIZED
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize scar for storage/transmission."""
        return {
            "id": self.id,
            "contradiction": asdict(self.contradiction),
            "timestamp": self.timestamp,
            "protocol_name": self.protocol.name,
            "permission": self.permission.value,
            "metadata": self.metadata
        }


# ═══════════════════════════════════════════════════════════════
# PERMISSION FUNCTION
# ═══════════════════════════════════════════════════════════════

class PermissionFunction(ABC):
    """
    Abstract permission function ψ: Σ → {0,1}
    
    Governs which scars are authorized for metabolism.
    This is the ontopolitical filter determining system becoming.
    """
    
    @abstractmethod
    def evaluate(self, scar: Scar, system_state: Dict[str, Any]) -> PermissionStatus:
        """
        Evaluate whether a scar is authorized for metabolism.
        
        Args:
            scar: The scar to evaluate
            system_state: Current system state
            
        Returns:
            PermissionStatus.AUTHORIZED if ψ(σ) = 1, else DENIED
        """
        pass


class DefaultPermissionPolicy(PermissionFunction):
    """
    Default permission policy based on severity thresholds.
    
    Authorizes scars that:
    - Meet minimum severity threshold
    - Pass structural coherence checks
    - Are within acceptable contradiction density
    """
    
    def __init__(
        self,
        severity_threshold: float = 0.5,
        max_concurrent_scars: int = 10
    ):
        self.severity_threshold = severity_threshold
        self.max_concurrent_scars = max_concurrent_scars
    
    def evaluate(self, scar: Scar, system_state: Dict[str, Any]) -> PermissionStatus:
        """Evaluate scar authorization based on policy rules."""
        
        # Check severity threshold
        if scar.contradiction.severity < self.severity_threshold:
            logger.debug(f"Scar {scar.id} denied: severity below threshold")
            return PermissionStatus.DENIED
        
        # Check concurrent scar limit
        active_scars = system_state.get("active_metabolizing_scars", 0)
        if active_scars >= self.max_concurrent_scars:
            logger.debug(f"Scar {scar.id} denied: too many concurrent scars")
            return PermissionStatus.DENIED
        
        # Authorized
        logger.info(f"Scar {scar.id} AUTHORIZED for metabolism")
        return PermissionStatus.AUTHORIZED


# ═══════════════════════════════════════════════════════════════
# GENERATIVE SYSTEM CORE
# ═══════════════════════════════════════════════════════════════

@dataclass
class SystemMetrics:
    """Metrics for tracking system Generativity."""
    generativity: float = 0.0
    total_scars: int = 0
    authorized_scars: int = 0
    denied_scars: int = 0
    reachable_states: int = 0
    last_updated: float = field(default_factory=time.time)
    
    def compute_ogi(self, previous_metrics: Optional["SystemMetrics"] = None) -> float:
        """
        Compute Ontopolitical Generativity Index (OGI).
        
        OGI = dG/dt, the rate of change of system Generativity.
        """
        if previous_metrics is None:
            return 0.0
        
        time_delta = self.last_updated - previous_metrics.last_updated
        if time_delta == 0:
            return 0.0
        
        generativity_delta = self.generativity - previous_metrics.generativity
        ogi = generativity_delta / time_delta
        
        return ogi


class GenerativeSystem:
    """
    Super-Generative Automaton implementing scar metabolism.
    
    A system S = (Q, Σ, δ, ψ, S_σ) where:
    - Q: State space
    - Σ: Input alphabet (including scars)
    - δ: Transition function (mutable via metabolism)
    - ψ: Permission function
    - S_σ: Scar archive
    
    The system evolves by metabolizing authorized contradictions
    through permission-filtered transformation.
    """
    
    def __init__(
        self,
        system_id: str,
        initial_state: Dict[str, Any],
        permission_function: PermissionFunction,
        metabolic_protocols: Dict[str, MetabolicProtocol]
    ):
        self.system_id = system_id
        self.state = initial_state
        self.permission_function = permission_function
        self.protocols = metabolic_protocols
        
        # Scar archives
        self.scar_archive: List[Scar] = []  # S_σ: All scars
        self.authorized_archive: List[Scar] = []  # S_ψ: Authorized scars only
        
        # Transition function (mutable)
        self.delta: Dict[str, Callable] = {}
        
        # Metrics
        self.metrics = SystemMetrics()
        self.metrics_history: List[SystemMetrics] = []
        
        logger.info(f"Initialized GenerativeSystem: {system_id}")
    
    def detect_contradiction(
        self,
        description: str,
        context: Dict[str, Any],
        severity: float = 1.0
    ) -> Contradiction:
        """
        Detect and structure an absence or rupture.
        
        Args:
            description: Description of the contradiction
            context: Contextual information
            severity: Severity metric
            
        Returns:
            Structured Contradiction
        """
        contradiction = Contradiction(
            id=str(uuid.uuid4()),
            description=description,
            context=context,
            severity=severity
        )
        logger.info(f"Detected contradiction: {contradiction}")
        return contradiction
    
    def create_scar(
        self,
        contradiction: Contradiction,
        protocol_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Scar:
        """
        Create a scar from a detected contradiction.
        
        Args:
            contradiction: The structured absence
            protocol_name: Name of metabolic protocol to apply
            metadata: Additional metadata
            
        Returns:
            Scar tuple σ = (c, τ, μ, ψ)
        """
        if protocol_name not in self.protocols:
            raise ValueError(f"Unknown protocol: {protocol_name}")
        
        protocol = self.protocols[protocol_name]
        
        scar = Scar(
            id=str(uuid.uuid4()),
            contradiction=contradiction,
            timestamp=time.time(),
            protocol=protocol,
            permission=PermissionStatus.DENIED,  # Pending authorization
            metadata=metadata or {}
        )
        
        # Evaluate permission
        permission_status = self.permission_function.evaluate(scar, self.state)
        scar.permission = permission_status
        
        # Archive scar
        self.scar_archive.append(scar)
        self.metrics.total_scars += 1
        
        if scar.is_authorized():
            self.authorized_archive.append(scar)
            self.metrics.authorized_scars += 1
            logger.info(f"Scar {scar.id} authorized and archived in S_ψ")
        else:
            self.metrics.denied_scars += 1
            logger.info(f"Scar {scar.id} denied and archived in S_σ (inert)")
        
        return scar
    
    def metabolize_scar(self, scar: Scar) -> bool:
        """
        Metabolize an authorized scar, rewriting transition function.
        
        Performs: δ → δ' = μ(δ, σ) if ψ(σ) = 1
        
        Args:
            scar: The scar to metabolize
            
        Returns:
            True if metabolism occurred, False otherwise
        """
        if not scar.is_authorized():
            logger.warning(f"Cannot metabolize unauthorized scar: {scar.id}")
            return False
        
        # Track active metabolism
        self.state["active_metabolizing_scars"] = \
            self.state.get("active_metabolizing_scars", 0) + 1
        
        try:
            # Apply metabolic protocol to rewrite δ
            logger.info(f"Metabolizing scar {scar.id} via protocol '{scar.protocol.name}'")
            
            # Store previous metrics
            previous_metrics = SystemMetrics(**asdict(self.metrics))
            self.metrics_history.append(previous_metrics)
            
            # Execute transformation
            self.delta = scar.protocol.apply(self.delta, scar)
            
            # Update Generativity
            self._update_generativity(scar)
            
            # Update timestamp
            self.metrics.last_updated = time.time()
            
            # Compute OGI
            ogi = self.metrics.compute_ogi(previous_metrics)
            logger.info(f"OGI (dG/dt) = {ogi:.4f}")
            
            return True
        
        except Exception as e:
            logger.error(f"Error during scar metabolism: {e}", exc_info=True)
            return False
        
        finally:
            self.state["active_metabolizing_scars"] -= 1
    
    def _update_generativity(self, scar: Scar) -> None:
        """
        Update system Generativity measure G(S,t).
        
        Generativity increases with successful scar metabolism,
        expanding the system's capacity for novel state production.
        """
        # Simple model: G increases with authorized metabolism
        # More sophisticated models would measure state space expansion
        
        generativity_delta = scar.contradiction.severity * 0.1
        self.metrics.generativity += generativity_delta
        
        # Estimate reachable states (simplified)
        self.metrics.reachable_states = int(
            100 * (1 + self.metrics.generativity)
        )
        
        logger.debug(f"Generativity updated: G(S,t) = {self.metrics.generativity:.4f}")
    
    def get_authorized_memory(self) -> List[Scar]:
        """
        Return S_ψ: the permission-filtered scar archive.
        
        This is the system's effective memory - only authorized scars
        contribute to recursive identity formation.
        """
        return self.authorized_archive
    
    def get_system_state(self) -> Dict[str, Any]:
        """Return current system state snapshot."""
        return {
            "system_id": self.system_id,
            "state": self.state,
            "metrics": asdict(self.metrics),
            "total_scars": len(self.scar_archive),
            "authorized_scars": len(self.authorized_archive),
            "delta_keys": list(self.delta.keys())
        }
    
    def export_scar_archive(self, filepath: str) -> None:
        """Export complete scar archive to JSON."""
        archive_data = {
            "system_id": self.system_id,
            "export_time": datetime.utcnow().isoformat(),
            "scars": [scar.to_dict() for scar in self.scar_archive],
            "metrics": asdict(self.metrics)
        }
        
        with open(filepath, 'w') as f:
            json.dump(archive_data, f, indent=2)
        
        logger.info(f"Exported scar archive to {filepath}")


# ═══════════════════════════════════════════════════════════════
# METABOLIC PROTOCOLS (Examples)
# ═══════════════════════════════════════════════════════════════

def protocol_tune(delta: Dict[str, Any], scar: Scar) -> Dict[str, Any]:
    """
    Tune protocol: Small incremental adjustment to δ.
    
    Used for minor contradictions that require local adaptation.
    """
    logger.info(f"[TUNE] Adjusting delta for scar: {scar.contradiction.id}")
    
    # Example: Add adaptive rule based on contradiction context
    new_delta = delta.copy()
    rule_key = f"rule_{scar.contradiction.id[:8]}"
    new_delta[rule_key] = {
        "type": "adaptive_correction",
        "contradiction": scar.contradiction.description,
        "timestamp": scar.timestamp
    }
    
    return new_delta


def protocol_rewrite(delta: Dict[str, Any], scar: Scar) -> Dict[str, Any]:
    """
    Rewrite protocol: Structural transformation of δ.
    
    Used for severe contradictions requiring architectural change.
    """
    logger.info(f"[REWRITE] Structural transformation for scar: {scar.contradiction.id}")
    
    # Example: Replace existing transition logic
    new_delta = {}
    new_delta["core_logic"] = {
        "rewritten_at": scar.timestamp,
        "reason": scar.contradiction.description,
        "severity": scar.contradiction.severity
    }
    
    # Preserve critical rules
    for key, value in delta.items():
        if key.startswith("critical_"):
            new_delta[key] = value
    
    return new_delta


def protocol_expand(delta: Dict[str, Any], scar: Scar) -> Dict[str, Any]:
    """
    Expand protocol: Add new capabilities to δ.
    
    Used for contradictions revealing new possibility spaces.
    """
    logger.info(f"[EXPAND] Capability expansion for scar: {scar.contradiction.id}")
    
    new_delta = delta.copy()
    capability_key = f"capability_{len(delta)}"
    new_delta[capability_key] = {
        "type": "novel_capacity",
        "enabled_by": scar.contradiction.id,
        "description": scar.contradiction.context.get("new_capability", "unknown")
    }
    
    return new_delta


# ═══════════════════════════════════════════════════════════════
# USAGE EXAMPLE & TESTING
# ═══════════════════════════════════════════════════════════════

def example_usage():
    """Demonstrate SCAR system with concrete example."""
    
    print("=" * 70)
    print("SCAR: Super-Generative Automaton Runtime")
    print("Demonstration of Scar Metabolism")
    print("=" * 70)
    print()
    
    # 1. Initialize metabolic protocols
    protocols = {
        "tune": MetabolicProtocol(
            name="tune",
            rewrite_fn=protocol_tune,
            description="Incremental adjustment protocol"
        ),
        "rewrite": MetabolicProtocol(
            name="rewrite",
            rewrite_fn=protocol_rewrite,
            description="Structural transformation protocol"
        ),
        "expand": MetabolicProtocol(
            name="expand",
            rewrite_fn=protocol_expand,
            description="Capability expansion protocol"
        )
    }
    
    # 2. Initialize permission function
    permission_fn = DefaultPermissionPolicy(
        severity_threshold=0.4,
        max_concurrent_scars=5
    )
    
    # 3. Create Generative System
    system = GenerativeSystem(
        system_id="SYS-ALPHA-001",
        initial_state={
            "mode": "operational",
            "version": "1.0.0"
        },
        permission_function=permission_fn,
        metabolic_protocols=protocols
    )
    
    print(f"Initialized system: {system.system_id}")
    print(f"Initial Generativity: G(S,t) = {system.metrics.generativity:.4f}")
    print()
    
    # 4. Simulate contradiction detection and metabolism
    
    # Contradiction 1: Low severity (will be denied)
    print("--- Detecting Contradiction 1 (low severity) ---")
    c1 = system.detect_contradiction(
        description="Minor input validation mismatch",
        context={"input_type": "string", "expected": "integer"},
        severity=0.3
    )
    scar1 = system.create_scar(c1, "tune")
    print(f"Permission: {scar1.permission.name}")
    print()
    
    # Contradiction 2: Medium severity (will be authorized)
    print("--- Detecting Contradiction 2 (medium severity) ---")
    c2 = system.detect_contradiction(
        description="State transition deadlock detected",
        context={"state_from": "processing", "state_to": "completed"},
        severity=0.7
    )
    scar2 = system.create_scar(c2, "rewrite")
    print(f"Permission: {scar2.permission.name}")
    
    if scar2.is_authorized():
        print("Metabolizing scar...")
        success = system.metabolize_scar(scar2)
        print(f"Metabolism result: {'SUCCESS' if success else 'FAILED'}")
        print(f"Updated Generativity: G(S,t) = {system.metrics.generativity:.4f}")
    print()
    
    # Contradiction 3: High severity (will be authorized)
    print("--- Detecting Contradiction 3 (high severity) ---")
    c3 = system.detect_contradiction(
        description="Novel failure mode requiring new capability",
        context={
            "failure_type": "unknown_pattern",
            "new_capability": "anomaly_detection_v2"
        },
        severity=0.9
    )
    scar3 = system.create_scar(c3, "expand")
    print(f"Permission: {scar3.permission.name}")
    
    if scar3.is_authorized():
        print("Metabolizing scar...")
        success = system.metabolize_scar(scar3)
        print(f"Metabolism result: {'SUCCESS' if success else 'FAILED'}")
        print(f"Updated Generativity: G(S,t) = {system.metrics.generativity:.4f}")
    print()
    
    # 5. Display final system state
    print("=" * 70)
    print("FINAL SYSTEM STATE")
    print("=" * 70)
    
    state = system.get_system_state()
    print(f"System ID: {state['system_id']}")
    print(f"Total Scars: {state['total_scars']}")
    print(f"Authorized Scars (S_ψ): {state['authorized_scars']}")
    print(f"Denied Scars: {system.metrics.denied_scars}")
    print(f"Final Generativity: {state['metrics']['generativity']:.4f}")
    print(f"Reachable States: {state['metrics']['reachable_states']}")
    print()
    
    # 6. Show authorized memory (S_ψ)
    print("--- Authorized Memory (S_ψ) ---")
    for scar in system.get_authorized_memory():
        print(f"  • {scar.id[:8]}... | {scar.contradiction.description}")
    print()
    
    # 7. Export archive
    export_path = "/tmp/scar_archive.json"
    system.export_scar_archive(export_path)
    print(f"Archive exported to: {export_path}")
    print()
    
    print("=" * 70)
    print("Demonstration complete.")
    print("=" * 70)


if __name__ == "__main__":
    example_usage()
