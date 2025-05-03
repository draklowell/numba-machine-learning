from genetic.crossover import Crossover
from genetic.mutation import Mutation
from nml import Tensor


class ChromosomePipeline:
    """
    A class that represents a pipeline for processing chromosome pairs.
    It applies a crossover operation followed by a mutation operation.
    """

    def __init__(self, crossover: Crossover, mutation: Mutation):
        self.crossover = crossover
        self.mutation = mutation

    def __call__(self, pairs: list[tuple[Tensor, Tensor]], ctx: dict) -> list[Tensor]:
        """
        Apply the pipeline to a list of pairs of tensors (chromosome pairs).
        """
        offspring = self.crossover(pairs, ctx)
        mutated_offspring = self.mutation(offspring, ctx)
        return mutated_offspring
