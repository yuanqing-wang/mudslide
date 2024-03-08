import random
from .generator import Generator

class Reaction(Generator):
    """Perform a reaction using crem. """
    @property
    def _str(self):
        return [
            "React",
            "Generate with reactions",
        ]
    
    def __str__(self):
        return random.choice(self._str)

    __repr__ = __str__

    def __call__(self) -> bool:
        return True

_all = [
    Reaction,
]

