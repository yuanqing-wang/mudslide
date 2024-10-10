import jax
import jax.numpy as jnp
from typing import Optional, NamedTuple
from .. import unit


class HarmonicAngleForce(NamedTuple):
    """Harmonic angle force."""

    particle1: Optional[jnp.ndarray] = None
    particle2: Optional[jnp.ndarray] = None
    particle3: Optional[jnp.ndarray] = None
    angle: Optional[jnp.ndarray] = None
    k: Optional[jnp.ndarray] = None

    if particle1 is None:
        particle1 = jnp.array([], dtype=jnp.int32)

    if particle2 is None:
        particle2 = jnp.array([], dtype=jnp.int32)

    if particle3 is None:
        particle3 = jnp.array([], dtype=jnp.int32)

    if angle is None:
        angle = jnp.array([], dtype=jnp.float64)

    if k is None:
        k = jnp.array([], dtype=jnp.float64)

    @classmethod
    def from_openmm(cls, force):
        """Initialize the force from an OpenMM HarmonicAngleForce object."""
        particle1, particle2, particle3, angle, k = [], [], [], [], []
        for idx in range(force.getNumAngles()):
            _particle1, _particle2, _particle3, _angle, _k = (
                force.getAngleParameters(idx)
            )
            particle1.append(_particle1)
            particle2.append(_particle2)
            particle3.append(_particle3)
            angle.append(_angle.value_in_unit(unit.ANGLE))
            k.append(_k.value_in_unit(unit.ENERGY / unit.ANGLE**2))
        return cls(
            particle1=jnp.array(particle1, dtype=jnp.int32),
            particle2=jnp.array(particle2, dtype=jnp.int32),
            particle3=jnp.array(particle3, dtype=jnp.int32),
            angle=jnp.array(angle, dtype=jnp.float64),
            k=jnp.array(k, dtype=jnp.float64),
        )

    def to_openmm(self):
        """Convert the force to an OpenMM HarmonicAngleForce object."""
        from openmm import HarmonicAngleForce as _HarmonicAngleForce

        force = _HarmonicAngleForce()
        for particle1, particle2, particle3, angle, k in zip(
            self.particle1, self.particle2, self.particle3, self.angle, self.k
        ):
            force.addAngle(
                int(particle1),
                int(particle2),
                int(particle3),
                float(angle),
                float(k),
            )
        return force

    def __call__(
        self,
        X: jnp.ndarray,
    ):
        X1 = X[self.particle1]
        X2 = X[self.particle2]
        X3 = X[self.particle3]
        left = X1 - X2
        right = X3 - X2
        angles = jnp.atan2(
            jnp.linalg.norm(jnp.cross(left, right), axis=-1),
            jnp.sum(left * right, axis=-1),
        )
        return 0.5 * self.k * (angles - self.angle) ** 2
