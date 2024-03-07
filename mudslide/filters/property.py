import abc
from ast import Not
from rdkit import Chem
from functools import partial

class Property(abc.ABC):
    """Base class for all properties. """
    @abc.abstractmethod
    def __call__(self, mol: Chem.Mol) -> bool:
        """Filter a molecule. """
        raise NotImplementedError
    
    @abc.abstractproperty
    def _str(self) -> list[str]:
        """Enumeate the string representation of the property. """
        raise NotImplementedError
    
    def __str__(self) -> str:
        return self._str[0]
    
    __repr__ = __str__

class NumberOf(Property):
    """Describe the number of elements in a molecule. 
    
    Parameters
    ----------
    element : str
        The element to be counted.

    Examples
    --------
    >>> import mudslide
    """
    def __init__(self, element: str):
        self.element = element

    @property
    def _element_number_mapping(self):
        return {
            "C": 6,
            "N": 7,
            "O": 8,
            "F": 9,
            "P": 15,
            "S": 16,
            "Cl": 17,
            "Br": 35,
            "I": 53,
        }

    @property
    def _element_name_mapping(self):
        return {
            "C": ["carbon", "carbons", "C"],
            "N": ["nitrogen", "nitrogens", "N"],
            "O": ["oxygen", "oxygens", "O"],
            "F": ["fluorine", "fluorines", "F"],
            "P": ["phosphorus", "phosphoruses", "P"],
            "S": ["sulfur", "sulfurs", "S"],
            "Cl": ["chlorine", "chlorines", "Cl"],
            "Br": ["bromine", "bromines", "Br"],
            "I": ["iodine", "iodines", "I"],
        }
    
    @property
    def _str(self):
        return [
            ("number of " + element) 
            for element in self._element_name_mapping[self.element]
        ]
    
    def __call__(self, mol: Chem.Mol) -> bool:
        return len(
            [
                atom for atom 
                in mol.GetAtoms() 
                if atom.GetAtomicNum() 
                == self._element_number_mapping[self.element]
            ]
        )
    
NumberOfCarbons = partial(NumberOf, "C")
NumberOfNitrogens = partial(NumberOf, "N")
NumberOfOxygens = partial(NumberOf, "O")
NumberOfFluorines = partial(NumberOf, "F")
NumberOfPhosphoruses = partial(NumberOf, "P")
NumberOfSulfurs = partial(NumberOf, "S")
NumberOfChlorines = partial(NumberOf, "Cl")
NumberOfBromines = partial(NumberOf, "Br")
NumberOfIodines = partial(NumberOf, "I")

