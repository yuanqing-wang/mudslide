from rdkit import Chem

def smilse2mol(fn):
    """Decorator to convert SMILES to RDKit mol objects. """
    def new_fn(*args, **kwargs):
        args = [
            Chem.MolFromSmiles(arg) 
            if isinstance(arg, str) else arg for arg in args
        ]
        kwargs = {
            k: Chem.MolFromSmiles(v) 
            if isinstance(v, str) else v for k, v in kwargs.items()
        }
        return fn(*args, **kwargs)
    new_fn.__name__ = fn.__name__
    new_fn.__doc__ = fn.__doc__
    return new_fn

