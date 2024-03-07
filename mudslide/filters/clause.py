import abc
import random
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

    Parameters
    ----------
    property : Property
        The property to be checked.

    relation : Relation
        The relation to be checked.

    value : Union[int, float]
        The value to be checked.

    Examples
    --------
    >>> from rdkit import Chem
    >>> from mudslide.filters.property import NumberOfCarbons
    >>> from mudslide.filters.relation import Eq
    >>> clause = SingleClause(NumberOfCarbons(), Eq(), 1)
    >>> clause(Chem.MolFromSmiles("C"))
    True
    >>> assert(isinstance(str(clause), str))
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
        return f"{self.property} {self.relation} {self.value}"
    
    __repr__ = __str__ 

class AndClause(Clause):
    """A clause describing the properties of a molecule.

    Parameters
    ----------
    clauses : list[Clause]
        A list of clauses.
    """
    def __init__(self, clauses: list[Clause]):
        self.clauses = clauses

    def __call__(self, mol) -> bool:
        return all(clause(mol) for clause in self.clauses)

    @property
    def _conjunctions(self):
        return [
            "and",
            ","
        ]

    def __str__(self) -> str:
        return f" {random.choice(self._conjunctions)} ".join(
            str(clause) for clause in self.clauses)
    
    __repr__ = __str__


