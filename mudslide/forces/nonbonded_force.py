from functools import partial
from dataclasses import dataclass
import jax
import jax.numpy as jnp
from typing import Optional, NamedTuple
import openmm
from .. import unit

PARTICLE = openmm.unit.mole.create_unit(
    6.02214076e23**-1,
    "particle",
    "particle",
)

COULOMB_CONSTANT_UNIT = (
    unit.ENERGY * unit.DISTANCE / ((openmm.unit.elementary_charge**2))
)

K_E = (
    8.9875517923
    * 1e9
    * openmm.unit.newton
    * openmm.unit.meter**2
    * openmm.unit.coulomb ** (-2)
    * PARTICLE ** (-1)
).value_in_unit(COULOMB_CONSTANT_UNIT)


@partial(jax.tree_util.register_dataclass, data_fields=["particle", "charge", "sigma", "epsilon"], meta_fields=[])
@dataclass
class NonbondedForce:
    """Nonbonded force."""
    particle: Optional[jnp.ndarray] = None
    charge: Optional[jnp.ndarray] = None
    sigma: Optional[jnp.ndarray] = None
    epsilon: Optional[jnp.ndarray] = None
    
    def __post_init__(self):
        if self.particle is None:
            self.particle = jnp.array([], dtype=jnp.int32)

        if self.charge is None:
            self.charge = jnp.array([], dtype=jnp.float64)

        if self.sigma is None:
            self.sigma = jnp.array([], dtype=jnp.float64)
        
        if self.epsilon is None:
            self.epsilon = jnp.array([], dtype=jnp.float64)
            
        charges = self.charge[:, None] * self.charge[None, :]
        sigmas = 0.5 * (self.sigma[:, None] + self.sigma[None, :])
        epsilons = (self.epsilon[:, None] * self.epsilon[None, :]) ** 0.5
        
        sigmas = jnp.fill_diagonal(sigmas, 0.0, inplace=False)
        epsilons = jnp.fill_diagonal(epsilons, 0.0, inplace=False)
        charges = jnp.fill_diagonal(charges, 0.0, inplace=False)
        
        self.charges = charges
        self.sigmas = sigmas
        self.epsilons = epsilons
        
    def __call__(self, X):
        """Compute the nonbonded energy."""
        X = X[self.particle]
        R = jax.nn.relu((X - X[:, None]) ** 2).sum(-1) ** 0.5
        R_inv = jnp.where(R > 0.0, 1.0 / R, 0.0)

        # calculating half of it because of the dubplicates
        u_coulomb = 0.5 * K_E * self.charges * R_inv
        u_lj = 2 * self.epsilons * ((self.sigmas * R_inv) ** 12 - (self.sigmas * R_inv) ** 6)
        energy = u_coulomb + u_lj
        return energy
    
    @classmethod
    def from_openmm(cls, force):
        """Initialize the nonbonded force from an OpenMM NonbondedForce object."""
        particle = []
        charge = []
        sigma = []
        epsilon = []
        for idx in range(force.getNumParticles()):
            params = force.getParticleParameters(idx)
            particle.append(idx)
            charge.append(params[0].value_in_unit(unit.CHARGE))
            sigma.append(params[1].value_in_unit(unit.DISTANCE))
            epsilon.append(params[2].value_in_unit(unit.ENERGY))
        
        return cls(
            particle=jnp.array(particle, dtype=jnp.int32),
            charge=jnp.array(charge, dtype=jnp.float64),
            sigma=jnp.array(sigma, dtype=jnp.float64),
            epsilon=jnp.array(epsilon, dtype=jnp.float64),
        )