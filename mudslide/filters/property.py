import abc
from ast import Not
import random
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
        return random.choice(self._str)
    
    __repr__ = __str__

class NumberOfElement(Property):
    """Describe the number of elements in a molecule. 
    
    Parameters
    ----------
    element : str
        The element to be counted.

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
    
NumberOfCarbons = partial(NumberOfElement, "C")
NumberOfNitrogens = partial(NumberOfElement, "N")
NumberOfOxygens = partial(NumberOfElement, "O")
NumberOfFluorines = partial(NumberOfElement, "F")
NumberOfPhosphoruses = partial(NumberOfElement, "P")
NumberOfSulfurs = partial(NumberOfElement, "S")
NumberOfChlorines = partial(NumberOfElement, "Cl")
NumberOfBromines = partial(NumberOfElement, "Br")
NumberOfIodines = partial(NumberOfElement, "I")

class NumberOfRings(Property):
    """Describe the number of rings in a molecule. """
    @property
    def _str(self):
        return [
            "number of rings",
            "number of ring",
            "number of cycles",
            "number of cycle",
            "ring",
            "rings",
            "cycle",
            "cycles",
        ]
    
    def __call__(self, mol: Chem.Mol) -> bool:
        return Chem.GetSSSR(mol)



