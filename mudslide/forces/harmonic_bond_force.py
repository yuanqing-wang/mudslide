from .force import Force
import jax
import jax.numpy as jnp
from typing import Optional

class HarmonicBondForce(Force):
    """Harmonic bond force. """
    particle1: Optional[jnp.ndarray] = None
    particle2: Optional[jnp.ndarray] = None
    length: Optional[jnp.ndarray] = None
    k: Optional[jnp.ndarray] = None

    if particle1 is None:
        particle1 = jnp.array([], dtype=jnp.int32)

    if particle2 is None:
        particle2 = jnp.array([], dtype=jnp.int32)

    if length is None:
        length = jnp.array([], dtype=jnp.float64)

    if k is None:
        k = jnp.array([], dtype=jnp.float64)

    def from_openmm(self, force):
        """Initialize the force from an OpenMM HarmonicBondForce object. """
        self.particle1 = jnp.array([bond[0] for bond in force.getBonds()])
        self.particle2 = jnp.array([bond[1] for bond in force.getBonds()])
        self.length = jnp.array([force.getBondParameters(i)[0] for i in range(force.getNumBonds())])
        self.k = jnp.array([force.getBondParameters(i)[1] for i in range(force.getNumBonds())])
    
