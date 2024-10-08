from curses import meta
from .integrator import Integrator
from functools import partial
from dataclasses import dataclass
import jax

@partial(jax.tree_util.register_dataclass, data_fields=["timestep"], meta_fields=[])
@dataclass
class VelocityVerletIntegrator(Integrator):
    timestep: float
    
    def step(self, state, system, key):
        position, velocity = state
        acceleration = self.acceleration(state, system)
        velocity = velocity + 0.5 * self.timestep * acceleration
        position = position + self.timestep * velocity
        state = state._replace(position=position, velocity=velocity)
        acceleration = self.acceleration(state, system)
        velocity = velocity + 0.5 * self.timestep * acceleration
        state = state._replace(velocity=velocity, position=position)
        return state
    