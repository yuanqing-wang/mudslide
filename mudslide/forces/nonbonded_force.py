from functools import partial
from dataclasses import dataclass
import numpy as onp
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
    sigma: Optional[jnp.ndarray] = None
    epsilon: Optional[jnp.ndarray] = None
    charge: Optional[jnp.ndarray] = None
    
    def __post_init__(self):
        if self.charge is None:
            self.charge = jnp.array([[]], dtype=jnp.float64)

        if self.sigma is None:
            self.sigma = jnp.array([[]], dtype=jnp.float64)
        
        if self.epsilon is None:
            self.epsilon = jnp.array([[]], dtype=jnp.float64)
            
        self.sigma, self.epsilon, self.charge = self.combine(
            self.sigma, self.epsilon, self.charge
        )
        
        self.sigma = jnp.fill_diagonal(self.sigma, 0.0, inplace=False)
        self.epsilon = jnp.fill_diagonal(self.epsilon, 0.0, inplace=False)
        self.charge = jnp.fill_diagonal(self.charge, 0.0, inplace=False)
        
    @staticmethod
    def combine(sigma, epsilon, charge):
        if sigma.ndim == 1:
            sigma = 0.5 * (sigma[:, None] + sigma[None, :])
        if epsilon.ndim == 1:
            epsilon = (epsilon[:, None] * epsilon[None, :]) ** 0.5
        if charge.ndim == 1:
            charge = charge[:, None] * charge[None, :]
        return sigma, epsilon, charge
        
    def __call__(self, X):
        """Compute the nonbonded energy."""
        R = jax.nn.relu((X - X[:, None]) ** 2).sum(-1) ** 0.5
        R_inv = jnp.where(R > 0.0, 1.0 / R, 0.0)

        # calculating half of it because of the dubplicates
        u_coulomb = 0.5 * K_E * self.charge * R_inv
        u_lj = 2 * self.epsilon * ((self.sigma * R_inv) ** 12 - (self.sigma * R_inv) ** 6)
        energy = u_coulomb + u_lj
        return energy
    
    @classmethod
    def from_openmm(cls, force):
        """Initialize the nonbonded force from an OpenMM NonbondedForce object."""
        charge = []
        sigma = []
        epsilon = []
        for idx in range(force.getNumParticles()):
            params = force.getParticleParameters(idx)
            charge.append(params[0].value_in_unit(unit.CHARGE))
            sigma.append(params[1].value_in_unit(unit.DISTANCE))
            epsilon.append(params[2].value_in_unit(unit.ENERGY))
        sigma, epsilon, charge = map(onp.array, (sigma, epsilon, charge))
        sigma, epsilon, charge = cls.combine(sigma, epsilon, charge)
        
        for idx in range(force.getNumExceptions()):
            i, j, chargeprod, _sigma, _epsilon = force.getExceptionParameters(idx)
            charge[j, i] = charge[i, j] = chargeprod.value_in_unit(unit.CHARGE ** 2)
            sigma[j, i] = sigma[i, j] = _sigma.value_in_unit(unit.DISTANCE)
            epsilon[j, i] = epsilon[i, j] = _epsilon.value_in_unit(unit.ENERGY)
        sigma, epsilon, charge = map(jnp.array, (sigma, epsilon, charge))
        
        return cls(sigma=sigma, epsilon=epsilon, charge=charge)