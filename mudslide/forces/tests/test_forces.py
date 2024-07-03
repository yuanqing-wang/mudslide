def test_harmonic_bond_force():
    import openmm
    _force = openmm.HarmonicBondForce()
    _force.addBond(0, 1, 1.0, 1.0)
    from mudslide.forces.harmonic_bond_force import HarmonicBondForce
    force = HarmonicBondForce.from_openmm(_force)
    __force = force.to_openmm()
    assert _force.getNumBonds() == __force.getNumBonds()

def test_harmonic_angle_force():
    import openmm
    _force = openmm.HarmonicAngleForce()
    _force.addAngle(0, 1, 2, 1.0, 1.0)
    from mudslide.forces.harmonic_angle_force import HarmonicAngleForce
    force = HarmonicAngleForce.from_openmm(_force)
    __force = force.to_openmm()
    assert _force.getNumAngles() == __force.getNumAngles()
