import numpy as onp

def test_system():
    import openmm as mm
    openmm_system = mm.System()
    openmm_system.addParticle(1.0)
    openmm_system.addParticle(1.0)
    force = mm.HarmonicBondForce()
    force.addBond(0, 1.0, 0.0, 1.0)
    openmm_system.addForce(force)
    
    
    from mudslide.system import System
    system = System.from_openmm(openmm_system)
    
    X = onp.random.randn(2, 3)
    mudslide_energy = system(X)
    
    from openmmtools.integrators import DummyIntegrator
    integrator = DummyIntegrator()
    context = mm.Context(openmm_system, integrator)
    context.setPositions(X)
    state = context.getState(getEnergy=True)
    openmm_energy = state.getPotentialEnergy().value_in_unit(mm.unit.kilojoule_per_mole)
    
    assert onp.allclose(mudslide_energy, openmm_energy)
    
    
    
    
    