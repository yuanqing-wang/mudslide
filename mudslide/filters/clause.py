import abc
from .property import Property
from .relation import Relation
from typing import Union

class Clause(abc.ABC):
    """Base class for all clauses. """
    @abc.abstractmethod
    def __call__(self, mol) -> bool:
        """Filter a molecule. """
        raise NotImplementedError
    
    def __str__(self) -> str:
        raise NotImplementedError
    
    __repr__ = __str__

class SingleClause(Clause):
    """A clause describing the properties of a molecule.


    """
    def __init__(
            self,
            property: Property,
            relation: Relation,
            value: Union[int, float],
    ):
        self.property = property
        self.relation = relation
        self.value = value

    def __call__(self, mol) -> bool:
        return self.relation(self.property(mol), self.value)

    def __str__(self) -> str:
        return f"{self.property} {self.relation}"
    
    __repr__ = __str__ 

class AndClause(Clause):
    """A clause describing the properties of a molecule.

    
    """


