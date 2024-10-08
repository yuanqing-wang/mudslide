import jax
from typing import NamedTuple, Optional
from . import unit
from .forces.harmonic_bond_force import HarmonicBondForce

class System(NamedTuple):
    masses: jax.numpy.ndarray
    forces: Optional[jax.numpy.ndarray] = None
    
    if forces is None:
        forces = jax.numpy.array([], dtype=jax.numpy.float64)
        
    def __call__(self, position):
        energy = 0.0
        for force in self.forces:
            energy = energy + force(position).sum()
        return energy
    
    @classmethod
    def from_openmm(cls, system):
        """Initialize the system from an OpenMM System object."""
        masses = []
        for idx in range(system.getNumParticles()):
            masses.append(system.getParticleMass(idx).value_in_unit(unit.MASS))
        
        forces = []
        for idx in range(system.getNumForces()):
            force = system.getForce(idx)
            print(force)
            if force.__class__.__name__ == "HarmonicBondForce":
                forces.append(HarmonicBondForce.from_openmm(force))
        
        return cls(masses=jax.numpy.array(masses, dtype=jax.numpy.float64), forces=forces)