from .generator import Generator

class Reaction(Generator):
    """Perform a reaction using crem. """
    def _str(self):
        return [
            "reaction",
        ]