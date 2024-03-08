import abc
import random

class Generator(abc.ABC):
    """Base class for all generators. """
    @abc.abstractmethod
    def __call__(self) -> str:
        """Generate a molecule. """
        raise NotImplementedError

    @abc.abstractproperty
    def _str(self) -> str:
        raise NotImplementedError
    
    def __str__(self):
        return random.choice(self._str)

    __repr__ = __str__