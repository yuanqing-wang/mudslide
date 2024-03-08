import random
from .generator import Generator

class Reaction(Generator):
    """Perform a reaction using crem. """
    def _str(self):
        return [
            "React",
            "Generate with reactions",
        ]
    
    def __str__(self):
        return random.choice(self._str)

_all = [
    Reaction,
]

