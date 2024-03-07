from .property import Property
from .relation import Relation

class Clause(object):
    """A clause describing the properties of a molecule.


    """
    def __init__(
            self,
            property: Property,
            relation: Relation,
    ):
        self.property = property
        self.relation = relation

    def __str__(self) -> str:
        return f"{self.property} {self.relation}"
    
    __repr__ = __str__
    
