import jax
from .integrator import Integrator

class BrownianIntegrator(Integrator):
    temperature: float
    frictionCoeff: float
    stepSize: float
    is_deterministic: bool = False

    def step(self, state, system, key):
        position, velocity = state
        epsilon = self.stepSize / (self.frictionCoeff * system.mass)
        force = self.force(state, system)
        noise = jax.random.normal(key, shape=position.shape)
        position = position + epsilon * force + (2 * epsilon * self.temperature) ** 0.5 * noise
        state = state._replace(position=position)
        return state
        
        
        
        

    