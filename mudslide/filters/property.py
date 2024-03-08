import abc
from ast import Not
import random
from rdkit import Chem
from rdkit.Chem import Lipinski
from functools import partial
from ..utils import smilse2mol

# =============================================================================
# Base class
# =============================================================================

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

    @abc.abstractclassmethod
    def sample(self):
        """Sample a property. """
        raise NotImplementedError
    
# =============================================================================
# Elements
# =============================================================================
    
ELEMENT_NAME_MAPPING = {
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

ELEMENT_NUMBER_MAPPING = {
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
    def _str(self):
        return [
            ("number of " + element) 
            for element in ELEMENT_NAME_MAPPING[self.element]
        ]
    
    @smilse2mol
    def __call__(self, mol: Chem.Mol) -> bool:
        return len(
            [
                atom for atom 
                in mol.GetAtoms() 
                if atom.GetAtomicNum() 
                == ELEMENT_NUMBER_MAPPING[self.element]
            ]
        )

    def sample(self):
        return random.randint(0, 10)
    
NumberOfCarbons = partial(NumberOfElement, "C")
NumberOfNitrogens = partial(NumberOfElement, "N")
NumberOfOxygens = partial(NumberOfElement, "O")
_all_elements = [
    NumberOfCarbons,
    NumberOfNitrogens,
    NumberOfOxygens,
]

# =============================================================================
# Lipinski
# =============================================================================

LIPINSKI_PROPERTIES = [
    "HeavyAtomCount",
    "NumAromaticRings",
    "NumHAcceptors",
    "NumHDonors",
    "NumHeteroatoms",
    "NumRotatableBonds",
    "RingCount",
]

LIPINSKI_PROPERTIES_NAME_MAPPING = {
    "HeavyAtomCount": [
        "heavy atom count",
        "heavy atom counts",
        "heavy atoms",
        "number of heavy atoms",
    ],
    "NumAromaticRings": [
        "number of aromatic rings",
        "aromatic ring count",
    ],
    "NumHAcceptors": [
        "number of H acceptors",
        "H acceptor count",
        "number of hydrogen acceptors",
        "hydrogen acceptor count",
        "number of hydrogen bond acceptors",
        "hydrogen bond acceptor count",
    ],
    "NumHDonors": [
        "number of H donors",
        "H donor count",
        "number of hydrogen donors",
        "hydrogen donor count",
        "number of hydrogen bond donors",
        "hydrogen bond donor count",
    ],
    "NumHeteroatoms": [
        "number of heteroatoms",
        "heteroatom count",
    ],
    "NumRotatableBonds": [
        "number of rotatable bonds",
        "rotatable bond count",
    ],
    "RingCount": [
        "ring count",
        "number of rings",
    ],
}

class LipinskiProperty(Property):
    """Describe the Lipinski properties of a molecule. 
    
    Parameters
    ----------
    property : str
        The Lipinski property to be checked.

    """
    def __init__(self, property: str):
        self.property = property
    
    @property
    def _str(self):
        return LIPINSKI_PROPERTIES_NAME_MAPPING[self.property]
    
    @smilse2mol
    def __call__(self, mol: Chem.Mol) -> bool:
        return getattr(Lipinski, self.property)(mol)
    
    def sample(self):
        return random.randint(0, 10)

_all_lipinski = []
for lipinski in LIPINSKI_PROPERTIES:
    globals()[lipinski] = partial(LipinskiProperty, lipinski)
    _all_lipinski.append(globals()[lipinski])

# =============================================================================
# Miscaleous
# =============================================================================
class MolecularWeight(Property):
    """Describe the molecular weight of a molecule. """
    @property
    def _str(self):
        return [
            "molecular weight",
            "weight",
            "molecular mass",
        ]
    
    @smilse2mol
    def __call__(self, mol: Chem.Mol) -> bool:
        return Chem.Descriptors.MolWt(mol)
    
    def sample(self):
        return random.randint(0, 500)
    
class TPSA(Property):
    """Describe the TPSA of a molecule. """
    @property
    def _str(self):
        return [
            "TPSA",
            "topological polar surface area",
        ]
    
    @smilse2mol
    def __call__(self, mol: Chem.Mol) -> bool:
        return Chem.rdMolDescriptors.CalcTPSA(mol)
    
    def sample(self):
        return random.randint(0, 200)
    
class LogP(Property):
    """Describe the logP of a molecule. """
    @property
    def _str(self):
        return [
            "logP",
            "partition coefficient",
        ]
    
    @smilse2mol
    def __call__(self, mol: Chem.Mol) -> bool:
        return Chem.Crippen.MolLogP(mol, addHs=True)
    
    def sample(self):
        return random.randint(-10, 10)

# =============================================================================
# All
# =============================================================================

_all = _all_elements + _all_lipinski

