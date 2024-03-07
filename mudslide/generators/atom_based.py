from .generator import Generator
import crem

class Grow(Generator):
    """Grow a molecule using crem. """
    def __call__(self, mol) -> str:
        return crem.grow(mol)
    
    def _str(self):
        return [
            "grow",
        ]
    
class Mutate(Generator):
    """Mutate a molecule using crem. """
    def __call__(self, mol) -> str:
        return crem.mutate(mol)
    
    def _str(self):
        return [
            "mutate",
        ]