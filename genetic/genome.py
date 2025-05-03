from genetic.chromosome import ChromosomePipeline
from genetic.selection import Selection
from nml import Tensor


class GenomePipeline:
    """
    A class that represents a pipeline for processing genomes.
    It applies a selection operation followed by a mutation operation.
    """

    selection: Selection
    pipelines: dict[str, ChromosomePipeline]
    elitarism_selection: Selection | None

    def __init__(
        self,
        selection: Selection,
        pipelines: dict[str, ChromosomePipeline],
        elitarism_selection: Selection | None = None,
    ):
        self.selection = selection
        self.pipelines = pipelines
        self.elitarism_selection = elitarism_selection

    def __call__(
        self, population: list[tuple[dict[str, Tensor], float]]
    ) -> list[dict[str, Tensor]]:
        """
        Apply the pipeline to a list of genomes (population).
        """
        selected = self.selection(population)
        elitarism_selected = []
        if self.elitarism_selection is not None:
            elitarism_selected = [
                genome for genome, _ in self.elitarism_selection(selected)
            ]

        # Sort from the best fitness to the worst
        selected.sort(key=lambda x: x[1], reverse=True)
        if len(selected) % 2 != 0:
            selected = selected[:-1]

        # Swap list -> dict to dict -> list
        chromosomes = {name: [] for name in self.pipelines}
        for genome, _ in selected:
            for name, chromosome in genome.items():
                chromosomes[name].append(chromosome)

        result = [{} for _ in range(len(selected) // 2)]
        # Apply crossover and mutations
        for name, genes in chromosomes.items():
            # Batch pairs
            pairs = []
            for i in range(0, len(selected), 2):
                parent1 = genes[i]
                parent2 = genes[i + 1]
                pairs.append((parent1, parent2))

            for i, tensor in enumerate(self.pipelines[name](pairs)):
                result[i][name] = tensor

        return result + elitarism_selected
