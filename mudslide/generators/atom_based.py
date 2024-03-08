from .generator import Generator
from typing import Optional
import os
from pathlib import Path
from crem import crem
import requests

DB_URL = """http://www.qsar4u.com/files/cremdb/replacements02_sc2.5.db.gz"""
DB_NAME = os.path.join(
    Path(__file__).parent.parent.parent, 
    ".cache/replacements02_sc2.5.db"
)

def _get_db():
    if not os.path.exists(DB_NAME):
        r = requests.get(DB_URL)
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
    >>> from mudslide.generators.atom_based import Grow
    >>> grow = Grow()
    >>> from rdkit import Chem
    >>> mol = Chem.MolFromSmiles("CCO")
    >>> smi = grow(mol)
    >>> print(smi)
    """
    def __call__(self, mol) -> str:
        return crem.grow_mol2(mol, db_name=self.db)
    
    def _str(self):
        return [
            "Grow",
        ]
    
class Mutate(CremGenerator):
    """Mutate a molecule using crem. 

    Examples
    --------
    >>> from mudslide.generators.atom_based import Grow
    >>> mutate = Mutate()
    >>> from rdkit import Chem
    >>> mol = Chem.MolFromSmiles("CCO")
    >>> smi = mutate(mol)
    """
    def __call__(self, mol) -> str:
        return crem.mutate_mol2(mol, db_name=self.db)
    
    def _str(self):
        return [
            "Mutate",
        ]