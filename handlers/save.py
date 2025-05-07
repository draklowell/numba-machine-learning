import pickle

from nml import Tensor, save_weights
from project.generation_handler import GenerationHandler


class SaveHandler(GenerationHandler):
    """
    A class that handles the generation callback of a genetic algorithm.
    It saves genomes.
    """

    path: str
    period: int

    def __init__(
        self,
        path: str,
        period: int,
    ):
        self.path = path
        self.period = period

    def on_generation(
        self,
        population: list[tuple[dict[str, Tensor]], float],
        labels: Tensor,
        generation: int,
        is_last: bool,
    ) -> bool:
        if generation % self.period == 0 or is_last:
            with open(self.path.format(generation=generation), "wb") as file:
                pickle.dump([(save_weights(x), y) for x, y in population], file)

        return False
