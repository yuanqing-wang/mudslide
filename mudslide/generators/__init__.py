from .generator import Generator
from .reaction_based import _all as _all_reaction
from .fragment_based import _all as _all_fragment
_all = _all_reaction + _all_fragment
import random
def sample():
    """Sample a generator. """
    return random.choice(_all)

def sample_reaction():
    """Sample a reaction-based generator. """
    return random.choice(_all_reaction)

def sample_fragment():
    """Sample a fragment-based generator. """
    return random.choice(_all_fragment)
