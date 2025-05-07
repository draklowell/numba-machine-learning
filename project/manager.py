import time
from warnings import warn

from genetic import GenomePipeline
from loader import DataManager
from nml import Device, Model, Sequential, Tensor
from project.fitness import FitnessEvaluator
from project.generation_handler import GenerationHandler


class Manager:
    """
    A class that manages the genetic algorithm process.

    Parameters:
        sequential: A Sequential model to be used in the genetic algorithm.
        fitness_evaluator: A FitnessEvaluator instance for evaluating model fitness.
        data_manager: A DataManager instance for managing data.
        genome_pipeline: A GenomePipeline instance for processing genomes.
        generation_handler: A function to handle the generation process.
        device: The device to run the models on (e.g., CPU or GPU).
        population_size: The size of the population for the genetic algorithm.
    """

    sequential: Sequential
    models: list[Model]
    fitness_evaluator: FitnessEvaluator
    data_manager: DataManager
    genome_pipeline: GenomePipeline
    generation_handler: GenerationHandler
    device: Device
    population_size: int
    last_generation: float

    def __init__(
        self,
        sequential: Sequential,
        fitness_evaluator: FitnessEvaluator,
        data_manager: DataManager,
        genome_pipeline: GenomePipeline,
        generation_handler: GenerationHandler,
        device: Device,
        population_size: int = 100,
    ):
        self.population_size = population_size
        self.sequential = sequential
        self.models = [sequential.build(device) for _ in range(population_size)]
        self.fitness_evaluator = fitness_evaluator
        self.data_manager = data_manager
        self.generation_handler = generation_handler
        self.genome_pipeline = genome_pipeline
        self.last_generation = 0

    def set_population(
        self, population: list[dict[str, Tensor]], replace: bool = False
    ):
        """
        Set the population of models.

        Parameters:
            population: A list of dictionaries representing weights of the population.
        """
        if len(population) != self.population_size and not replace:
            raise ValueError(
                f"Population size {len(population)} does not match the expected size {self.population_size}."
            )

        for model, weights in zip(self.models, population):
            model.replace_weights(weights)

    def run_generation(self, generation: int, is_last: bool) -> bool:
        """
        Run a single generation of the genetic algorithm.

        1. Get data from the data manager.
        2. For each model in the population:
            - Get model predictions.
            - Evaluate fitness using the fitness evaluator.
            - Generate genome from the model.
        3. Call the on_inferences method with the population.
        4. Generate a new population using the genome pipeline.
        5. Replace the weights of each model with the new genome.

        Parameters:
            generation: The current generation number.
            is_last: Whether this is the last generation.
        """
        profile = {"start": time.time(), "last_generation": self.last_generation}
        # Get data
        images, labels = self.data_manager()

        population = []
        for model in self.models:
            # Get model predictions
            predictions = model(images)

            # Generate genome
            genome = model.get_weights()
            population.append((genome, predictions))

        for idx, (genome, predictions) in enumerate(population):
            # Evaluate fitness
            fitness = self.fitness_evaluator(predictions.wait(), labels)
            population[idx] = (genome, fitness)

        profile["fitness"] = time.time()

        if (
            self.generation_handler.on_generation(
                population, labels, generation, is_last
            )
            or is_last
        ):
            return True

        new_population = self.genome_pipeline(population)
        if len(new_population) > self.population_size:
            new_population = new_population[: self.population_size]

            warn(
                "The population size is smaller than the number of genomes generated. "
                "Some genomes will be discarded."
            )
        elif len(new_population) < self.population_size:
            warn(
                "The population size is larger than the number of genomes generated. "
                "Some genomes will be duplicated."
            )
            new_population += new_population[
                : self.population_size - len(new_population)
            ]

        for genome, model in zip(new_population, self.models):
            model.replace_weights(genome)

        profile["pipeline"] = time.time()
        self.generation_handler.on_profile(profile, generation)
        self.last_generation = time.time()

        return False

    def run(self, max_generations: int = 1000):
        """
        Run the genetic algorithm for a specified number of generations.

        Parameters:
            max_generations: The maximum number of generations to run.
        """
        self.last_generation = time.time()
        for generation in range(max_generations):
            if self.run_generation(generation, generation == max_generations - 1):
                return
