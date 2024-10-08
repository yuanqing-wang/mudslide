import jax
import jax.numpy as jnp
from typing import Optional, NamedTuple
from .. import unit

class HarmonicBondForce(NamedTuple):
    """Harmonic bond force."""

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
        
    @classmethod
    def from_openmm(cls, force):
        """Initialize the force from an OpenMM HarmonicBondForce object."""
        particle1, particle2, length, k = [], [], [], []
        for idx in range(force.getNumBonds()):
            _particle1, _particle2, _length, _k = force.getBondParameters(idx)
            particle1.append(_particle1)
            particle2.append(_particle2)
            length.append(_length.value_in_unit(unit.DISTANCE))
            k.append(_k.value_in_unit(unit.ENERGY / unit.DISTANCE**2))
        return cls(
            particle1=jnp.array(particle1, dtype=jnp.int32),
            particle2=jnp.array(particle2, dtype=jnp.int32),
            length=jnp.array(length, dtype=jnp.float64),
            k=jnp.array(k, dtype=jnp.float64),
        )

    def to_openmm(self):
        """Convert the force to an OpenMM HarmonicBondForce object."""
        from openmm import HarmonicBondForce as _HarmonicBondForce

        force = _HarmonicBondForce()
        for particle1, particle2, length, k in zip(
            self.particle1, self.particle2, self.length, self.k
        ):
            force.addBond(
                int(particle1), 
                int(particle2), 
                float(length) * unit.DISTANCE, 
                float(k) * unit.ENERGY / unit.DISTANCE**2,
            )
        return force

    def __call__(self, X: jnp.ndarray):
        X1 = X[self.particle1]
        X2 = X[self.particle2]
        deltaX = X2 - X1
        distance = (deltaX **2 ).sum(-1) ** 0.5
        return 0.5 * self.k * (distance - self.length) ** 2
