import jax
import jax.numpy as jnp
from typing import Optional, NamedTuple
from .. import unit

class NonbondedForce(NamedTuple):
    """Nonbonded force."""
    particle: Optional[jnp.ndarray] = None
    charge: Optional[jnp.ndarray] = None
    sigma: Optional[jnp.ndarray] = None
    epsilon: Optional[jnp.ndarray] = None

    if particle is None:
        particle = jnp.array([], dtype=jnp.int32)

    if charge is None:
        charge = jnp.array([], dtype=jnp.float64)

    if sigma is None:
        sigma = jnp.array([], dtype=jnp.float64)