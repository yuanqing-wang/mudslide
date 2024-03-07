import abc

class Generator(abc.ABC):
    """Base class for all generators. """
    @abc.abstractmethod
    def __call__(self) -> str:
        """Generate a molecule. """
        raise NotImplementedError

    @abc.abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError

    __repr__ = __str__