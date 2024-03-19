import random
from typing import Optional
import os
from pathlib import Path
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from crem import crem
import requests
from .generator import Generator
from ..utils import smilse2mol

DB_URL = """http://www.qsar4u.com/files/cremdb/replacements02_sc2.5.db.gz"""
DB_NAME = os.path.join(
    Path(__file__).parent.parent.parent, 
    ".cache/replacements02_sc2.5.db"
)

def _get_db():
    if not os.path.exists(DB_NAME):
        r = requests.get(DB_URL)
        os.mkdir(os.path.dirname(DB_NAME))
        with open(f"{DB_NAME}.gz", "wb") as f:
            f.write(r.content)
        os.system(f"gzip -dk {DB_NAME}.gz")
    return DB_NAME

class CremGenerator(Generator):
    """A generator that uses the CReM library. """
    def __init__(self, db: Optional[str]=None):
        if db is None:
            db = _get_db()
        self.db = db
        
class Grow(CremGenerator):
    """Grow a molecule using crem. 
    
    Examples
    --------
    >>> from mudslide.generators.fragment_based import Grow
    >>> grow = Grow()
    >>> from rdkit import Chem
    >>> mol = Chem.MolFromSmiles("CCO")
    >>> smi = grow(mol)
    >>> print(smi)
    """
    @smilse2mol
    def __call__(self, mol) -> str:
        return random.choice(crem.grow_mol2(mol, db_name=self.db))
    
    @property
    def _str(self):
        return [
            "Grow",
        ]
    
class Mutate(CremGenerator):
    """Mutate a molecule using crem. 

    Examples
    --------
    >>> from mudslide.generators.fragment_based import Grow
    >>> mutate = Mutate()
    >>> from rdkit import Chem
    >>> mol = Chem.MolFromSmiles("CCO")
    >>> smi = mutate(mol)
    """
    @smilse2mol
    def __call__(self, mol) -> str:
        mutated = crem.mutate_mol2(mol, db_name=self.db)
        if len(mutated):
            return random.choice(mutated)
    
    @property
    def _str(self):
        return [
            "Mutate",
        ]
    
_all = [
    Grow,
    Mutate,
]
