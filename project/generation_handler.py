from nml import Tensor


class GenerationHandler:
    """
    A base class for handling generation callbacks in a genetic algorithm.
    This class defines the interface for handling generation events.
    """

    def on_generation(
        self,
        population: list[tuple[dict[str, Tensor]], float],
        labels: Tensor,
        generation: int,
        is_last: bool,
    ) -> bool:
        """
        Called at the end of each fitness evaluation.

        Parameters:
            population: The population of genomes and their fitness scores.
            labels: The labels for the current generation.
            generation: The current generation number.
            is_last: Whether this is the last generation.

        Returns:
            True if the genetic algorithm should be stopped, False otherwise.
        """
        return False

    def on_profile(
        self,
        profile: dict[str, float],
        generation: int,
    ) -> None:
        """
        Called at the end of each generation.

        Parameters:
            profile: A dictionary containing profiling information.
            generation: The current generation number.
        """
