from tkinter import SEPARATOR
from typing import Optional, Callable
from .generators import Generator, sample_fragment, sample_reaction
from .generators.fragment_based import CremGenerator
from .filters import Clause, sample as sample_clause

CONJUNCTION = "with"
SEPARATOR = "<SEP>"


def combine(
        generator: str,
        fro: str,
        to: str,
        clause: Optional[Clause] = None,
) -> str:
    """Combine the generator, fro, to, and clause into a sentence. 
    
    Parameters
    ----------
    generator : str
        The name of the generator.

    fro : str
        The starting molecule.

    to : str
        The ending molecule.

    clause : Optional[Clause]
        A clause that describes the transformation.

    Returns
    -------
    str
        A sentence that describes the transformation.

    Examples
    --------
    >>> from mudslide.sentence import combine
    >>> combine("Grow", "CCO", "CCCO")
    'Grow CCO<SEP>CCCO'
    >>> combine("Mutate", "CCO", "CCCO", "with number of carbons smaller than 4")
    'Mutate CCO with with number of carbons smaller than 4<SEP>CCCO'
    """
    if clause is not None:
        clause = " " + CONJUNCTION + " " + str(clause)
    else:
        clause = ""
    return f"{generator} {fro}{clause}{SEPARATOR}{to}"
    
class Sentence(object):
    """Base class for all sentences. """
    def __init__(
            self,
            generator: Generator,
            fro: str,
            to: Optional[str] = None,
            clause: Optional[Clause] = None,
    ):
        self.generator = generator
        self.fro = fro
        self.to = to
        self.clause = clause

        if self.to is None:
            # if to is not given, then we need to generate it 
            # using the atom-based generator
            assert isinstance(
                self.generator, CremGenerator
            ), "Only crem generator can have no `to`."

            self.to = str(self.generator(self.fro))

    def __str__(self) -> str:
        return combine(
            str(self.generator), 
            self.fro, 
            self.to, 
            self.clause,
        )
    
    __repr__ = __str__

    def __call__(self) -> bool:
        if self.clause is not None:
            return self.clause(self.to)
        return True
    
def sample(
        fro: str,
        to: Optional[str] = None,
):
    """Sample a sentence. 
    
    Parameters
    ----------
    to : str
        The ending molecule.

    fro : Optional[str]
        The starting molecule. If not given, then it is randomly sampled.

    Returns
    -------
    str
        A sentence that describes the transformation.

    """
    # sample a generator
    if to is None:
        generator = sample_fragment()()
        to = generator(fro)
    else:
        generator = sample_reaction()()

    # sample a clause
    clause = sample_clause(to)

    # create a sentence
    sentence = Sentence(generator, fro, to, clause)

    return sentence

