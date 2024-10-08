import abc
import jax
from dataclasses import dataclass

class Integrator:
    @abc.abstractmethod
    def step(self, state, system, key):
        raise NotImplementedError
    
    def force(self, state, system):
        position, _ = state        
        return -jax.grad(system)(position)
    
    def acceleration(self, state, system):
        force = self.force(state, system)
        return force / system.masses[..., None]

    def __call__(self, steps, state, system, key):
        keys = jax.random.split(key, steps)
        state = jax.lax.fori_loop(
            0, steps, lambda i, state: self.step(state, system, keys[i]), state,
        )
        return state
        
        