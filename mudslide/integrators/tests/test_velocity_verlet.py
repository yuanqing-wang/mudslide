import jax
import jax.numpy as jnp
import numpy as onp

# def test_velocity_verlet():
#     from mudslide.integrators.velocity_verlet import VelocityVerletIntegrator
#     from mudslide.system import System
#     from mudslide.state import State
#     from mudslide.forces.harmonic_bond_force import HarmonicBondForce
#     force = HarmonicBondForce(
#         particle1=jnp.array([0]), 
#         particle2=jnp.array([1]), 
#         length=jnp.array([0.0]), 
#         k=jnp.array([1.0]),
#     )
#     system = System(masses=jnp.array([1.0, 1.0]), forces=[force])
#     integrator = VelocityVerletIntegrator(timestep=0.1)
#     positions = jax.random.normal(jax.random.PRNGKey(0), shape=(2, 3))
#     velocities = jax.random.normal(jax.random.PRNGKey(1), shape=(2, 3))
#     state = State(positions, velocities)
#     state = integrator.step(state, system, key=0)

def test_openmm_consistency():    
    X = onp.random.randn(2, 3)
    V = onp.random.randn(2, 3)
    
    # create an OpenMM system with bonded force
    import openmm as mm
    system = mm.System()
    system.addParticle(1.0)
    system.addParticle(1.0)
    force = mm.HarmonicBondForce()
    force.addBond(0, 1, 0.0, 1.0)
    system.addForce(force)

    from openmmtools.integrators import VelocityVerletIntegrator
    integrator = VelocityVerletIntegrator(0.1 * mm.unit.picosecond)
    context = mm.Context(system, integrator)
    context.setPositions(X * mm.unit.nanometer)
    context.setVelocities(V * mm.unit.nanometer/mm.unit.picosecond)
    integrator.step(1)
    X_openmm = context.getState(getPositions=True).getPositions(asNumpy=True) / mm.unit.nanometer
    V_openmm = context.getState(getVelocities=True).getVelocities(asNumpy=True) / mm.unit.nanometer*mm.unit.picosecond
    
    
    from mudslide.integrators.velocity_verlet import VelocityVerletIntegrator
    from mudslide.system import System
    from mudslide.state import State
    from mudslide.forces.harmonic_bond_force import HarmonicBondForce
    force = HarmonicBondForce(
        particle1=jnp.array([0]), 
        particle2=jnp.array([1]), 
        length=jnp.array([0.0]), 
        k=jnp.array([1.0]),
    )
    system = System(masses=jnp.array([1.0, 1.0]), forces=[force])
    integrator = VelocityVerletIntegrator(timestep=0.1)
    state = State(jnp.array(X), jnp.array(V))
    state = integrator(1, state, system, key=jax.random.PRNGKey(0))
    X_mudslide, V_mudslide = state
    
    print((V_mudslide - V) / (V_openmm - V))
    # print((X_mudslide - X) / (X_openmm - X))
        
    assert jnp.allclose(X_openmm, X_mudslide, atol=1e-3)
    assert jnp.allclose(V_openmm, V_mudslide, atol=1e-1)
    

    
    
