import jax
import jax.numpy as jnp
from typing import Optional, NamedTuple
from .. import unit


class PeriodicTorsionForce(NamedTuple):
    """Periodic torsion force."""

    particle1: Optional[jnp.ndarray] = None
    particle2: Optional[jnp.ndarray] = None
    particle3: Optional[jnp.ndarray] = None
    particle4: Optional[jnp.ndarray] = None
    phase: Optional[jnp.ndarray] = None
    k: Optional[jnp.ndarray] = None
    periodicity: Optional[jnp.ndarray] = None

    if particle1 is None:
        particle1 = jnp.array([], dtype=jnp.int32)

    if particle2 is None:
        particle2 = jnp.array([], dtype=jnp.int32)

    if particle3 is None:
        particle3 = jnp.array([], dtype=jnp.int32)

    if particle4 is None:
        particle4 = jnp.array([], dtype=jnp.int32)

    if phase is None:
        phase = jnp.array([], dtype=jnp.float64)

    if k is None:
        k = jnp.array([], dtype=jnp.float64)

    if periodicity is None:
        periodicity = jnp.array([], dtype=jnp.int32)

    @classmethod
    def from_openmm(cls, force):
        """Initialize the force from an OpenMM PeriodicTorsionForce object."""
        particle1, particle2, particle3, particle4, periodicity, phase, k = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for idx in range(force.getNumTorsions()):
            (
                _particle1,
                _particle2,
                _particle3,
                _particle4,
                _periodicity,
                _phase,
                _k,
            ) = force.getTorsionParameters(idx)
            particle1.append(_particle1)
            particle2.append(_particle2)
            particle3.append(_particle3)
            particle4.append(_particle4)
            phase.append(_phase.value_in_unit(unit.ANGLE))
            k.append(_k.value_in_unit(unit.ENERGY))
            periodicity.append(_periodicity)
        return cls(
            particle1=jnp.array(particle1, dtype=jnp.int32),
            particle2=jnp.array(particle2, dtype=jnp.int32),
            particle3=jnp.array(particle3, dtype=jnp.int32),
            particle4=jnp.array(particle4, dtype=jnp.int32),
            periodicity=jnp.array(periodicity, dtype=jnp.int32),
            phase=jnp.array(phase, dtype=jnp.float64),
            k=jnp.array(k, dtype=jnp.float64),
        )

    def to_openmm(self):
        """Convert the force to an OpenMM PeriodicTorsionForce object."""
        from openmm import PeriodicTorsionForce as _PeriodicTorsionForce

        force = _PeriodicTorsionForce()
        for (
            particle1,
            particle2,
            particle3,
            particle4,
            periodicity,
            phase,
            k,
        ) in zip(
            self.particle1,
            self.particle2,
            self.particle3,
            self.particle4,
            self.periodicity,
            self.phase,
            self.k,
        ):
            force.addTorsion(
                int(particle1),
                int(particle2),
                int(particle3),
                int(particle4),
                int(periodicity),
                float(phase),
                float(k),
            )
        return force

    def energy(
        self,
        X: jnp.ndarray,
    ):
        """Compute the energy."""
        # compute displacements 0->1, 2->1, 2->3
        r01 = X[self.particle2] - X[self.particle1]
        r21 = X[self.particle2] - X[self.particle3]
        r23 = X[self.particle4] - X[self.particle3]

        # compute normal planes
        n1 = jnp.cross(r01, r21)
        n2 = jnp.cross(r21, r23)

        rkj_normed = r21 / jnp.linalg.norm(r21, axis=-1, keepdims=True)

        y = jnp.sum(jnp.cross(n1, n2) * rkj_normed, axis=-1)
        x = jnp.sum(n1 * n2, axis=-1)

        # choose quadrant correctly
        phi = jnp.atan2(y, x)

        # compute energy
        energy = self.k * (1 + jnp.cos(self.periodicity * phi - self.phase))
        return energy
