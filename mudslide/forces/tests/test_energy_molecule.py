import openmm as mm
import numpy as onp 

def test_energy_molecule():
    from openff.toolkit.topology import Molecule
    from openff.toolkit.typing.engines.smirnoff import ForceField
    ff = ForceField("openff-2.0.0.offxml")
    molecule = Molecule.from_smiles("Cn1cnc2c1c(=O)n(C)c(=O)n2C")
    # molecule = Molecule.from_smiles("CCO")
    molecule.generate_conformers(n_conformers=1)
    
    mm_system = ff.create_openmm_system(molecule.to_topology())
    import mudslide as md
    from mudslide.system import System
    md_system = System.from_openmm(mm_system)
    
    from openmmtools.integrators import DummyIntegrator
    from openff.units import unit as off_unit
    X = onp.random.randn(molecule.n_atoms, 3)

    
    integrator = DummyIntegrator()
    simulation = mm.app.Simulation(molecule.to_topology().to_openmm(), mm_system, integrator)
    simulation.context.setPositions(X)
    state = simulation.context.getState(getEnergy=True)
    
    
    openmm_energy = state.getPotentialEnergy().value_in_unit(mm.unit.kilojoule_per_mole)
    # break down the energy into components
    
    # force_types = [mm.HarmonicBondForce, mm.HarmonicAngleForce, mm.PeriodicTorsionForce, mm.NonbondedForce]
    # force_2_idx = {force: i for i, force in enumerate(force_types)}
    # force_2_energy = {}
    
    # Assign force groups to each force type
    # for i in range(mm_system.getNumForces()):
    #     force = mm_system.getForce(i)
    #     for force_type in force_types:
    #         if isinstance(force, force_type):
    #             force.setForceGroup(force_2_idx[force_type])
    
    # for i in range(mm_system.getNumForces()):
    #     force = mm_system.getForce(i)
        
    #     for force_type in force_types:
    #         if isinstance(force, force_type):
    #             state = simulation.context.getState(getEnergy=True, groups={force_2_idx[force_type]})
    #             energy = state.getPotentialEnergy()
    #             force_2_energy[force_type.__name__] = energy.value_in_unit(mm.unit.kilojoule_per_mole)
                
            
    mudslide_energy = md_system(X)
    
    
    assert onp.allclose(mudslide_energy, openmm_energy)