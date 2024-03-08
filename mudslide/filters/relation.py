import abc
from typing import Union
import random

class Relation(abc.ABC):
    """Base class for all relations. """
    @abc.abstractmethod
    def __call__(
        self,
        x: Union[int, float],
        y: Union[int, float]
    ) -> bool:
        """Filter a pair of molecules. """
        raise NotImplementedError
    
    @abc.abstractproperty
    def _str(self) -> list[str]:
        """Enumeate the string representation of the relation. """
        raise NotImplementedError
    
    def __str__(self) -> str:
        return random.choice(self._str)
    
    __repr__ = __str__
    
class Eq(Relation):
    """Filter a pair of molecules. """
    def __call__(
        self,
        x: Union[int, float],
        y: Union[int, float]
    ) -> bool:
        return x == y
    
    @property
    def _str(self) -> list[str]:
        return [
            "=",
            "==",
            "equal",
            "equals",
            "is",
            "are",
            "being",
        ]
    
class Gt(Relation):
    """Filter a pair of molecules. """
    def __call__(
        self,
        x: Union[int, float],
        y: Union[int, float]
    ) -> bool:
        return x > y
    
    @property
    def _str(self) -> list[str]:
        return [
            ">",
            "greater than",
            "higher than",
            "more than",
            "exceeds",
            "exceeding",
            "over",
            "above",
        ]
    
class Lt(Relation):
    """Filter a pair of molecules. """
    def __call__(
        self,
        x: Union[int, float],
        y: Union[int, float]
    ) -> bool:
        return x < y
    
    @property
    def _str(self) -> list[str]:
        return [
            "<",
            "less than",
            "lower than",
            "fewer than",
            "under",
            "below",
        ]


    
