import abc
import random
from .property import Property, _all as _all_property
from .relation import Relation, Eq, Gt, Lt
_all_relation = [Eq, Gt, Lt]
from ..utils import smilse2mol
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
            " and ",
            ", "
        ]

    def __str__(self) -> str:
        return f"{random.choice(self._conjunctions)}".join(
            str(clause) for clause in self.clauses)
    
    __repr__ = __str__

@smilse2mol
def sample_once(mol, max_iter: int=10) -> SingleClause:
    """Sample a single clause from a molecule.

    Parameters
    ----------
    mol : str
        The molecule to be checked.

    Returns
    -------
    str
        A clause that describes the molecule.
    """
    for _ in range(max_iter):
        # choose a property
        property = random.choice(_all_property)()

        # choose a relation
        relation = random.choice(_all_relation)()

        # choose a value
        value = property.sample()

        # create a clause
        clause = SingleClause(property, relation, value)

        # if the clause stands, return
        if clause(mol):
            return clause
    return None


def sample(mol, max_iter: int=10, max_samples: int=5) -> Clause:
    """Sample multiple clauses.

    Parameters
    ----------
    mol : str
        The molecule to be checked.

    max_iter : int
        The maximum number of iterations to sample a single clause.

    max_samples : int
        The maximum number of clauses to be sampled.
    """
    num_samples = random.randint(1, max_samples)
    clauses = []
    for _ in range(num_samples):
        clause = sample_once(mol, max_iter)
        if clause is not None:
            clauses.append(clause)
    if len(clauses) == 0:
        return None
    return AndClause(clauses)




