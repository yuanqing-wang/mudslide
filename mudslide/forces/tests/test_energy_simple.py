def test_harmonic_bond_force_energy():
    import openmm
    import jax.numpy as jnp
    _force = openmm.HarmonicBondForce()
    _force.addBond(0, 1, 2.0, 1.0)
    from mudslide.forces.harmonic_bond_force import HarmonicBondForce
    force = HarmonicBondForce.from_openmm(_force)
    X = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    energy = force.energy(X)
    assert energy == 0.5

def test_harmonic_angle_force_energy():
    import openmm
    import jax.numpy as jnp
    import math
    _force = openmm.HarmonicAngleForce()
    _force.addAngle(0, 1, 2, math.pi / 4.0, 1.0)
    from mudslide.forces.harmonic_angle_force import HarmonicAngleForce
    force = HarmonicAngleForce.from_openmm(_force)
    X = jnp.array([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    energy = force.energy(X)
    assert jnp.allclose(energy, 0.5 * (math.pi / 4.0) ** 2)

def test_periodic_torsion_force_energy():
    import openmm
    import jax.numpy as jnp
    import math
    _force = openmm.PeriodicTorsionForce()
    _force.addTorsion(0, 1, 2, 3, 1, math.pi / 4.0, 1.0)
    from mudslide.forces.periodic_torsion_force import PeriodicTorsionForce
    force = PeriodicTorsionForce.from_openmm(_force)
    X = jnp.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    energy = force.energy(X)
    assert jnp.allclose(
        energy,
        (1.0 + jnp.cos(jnp.pi / 2.0 - math.pi / 4.0)),
    )