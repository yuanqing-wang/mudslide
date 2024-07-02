import abc
from typing import NamedTuple

class Force(NamedTuple):
    """Base class for all forces in the simulation. """

    @abc.abstractmethod
    def from_openmm(self, force):
        """Initialize the force from an OpenMM force object. """
        raise NotImplementedError
    
    @abc.abstractmethod
    def to_openmm(self):
        """Return an OpenMM force object. """
        raise NotImplementedError
    
