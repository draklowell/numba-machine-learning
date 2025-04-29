"""
assemblance of all the functions of genetic algo
"""
import crossover as cross
import selection as select
import mutation as mut

class Genetic:
    """
    Manage genetic algorithm
    """
    CHOICES = {
        'selection':{
            'tournament': select.tournament,
            'roulette' : select.roulette,
            'rank' : select.rank
        },
        'crossover':{
            'single_point':cross.single_point,
            'two_point':cross.two_point,
            'uniform':cross.uniform
        }
    }

    def __init__(self, limitations:list, selection_type:str, crossover_type:str):
        """
        You may use different selection and corssover algoritms.

        :param limitations: list, limitations for mutations.
        :param selection_type: str, may be one of (tournament, roulette, rank).
        :param crossover_type: str, may be one of (single_point, two_point, uniform).
        """
        if selection_type not in Genetic.CHOICES['selection']:
            raise ValueError(f"Bad input for selection_type. Must be one of: \
{tuple(x for x in Genetic.CHOICES['selection'])}")
        if crossover_type not in Genetic.CHOICES['crossover']:
            raise ValueError(f"Bad input for selection_type. Must be one of: \
{tuple(x for x in Genetic.CHOICES['crossover'])}")

        self.limitations = limitations
        self.selection_type = selection_type
        self.crossover_type = crossover_type

    def selection(self, candidates:list[tuple[list, int]], size, tournament_size=2):
        """
        Selects gens to be passed based on fitness and in a way that was selected

        :param candidates: list, candidates among witch to select.
        :param size: int, number of best gens to be selected.
        :param tournament_size: int, number of gens among witch to select during\
             one tournament. Default 2
        :return: list, gens selected among given.
        """
        args = [candidates, size] if self.selection_type != 'tournament' \
            else [candidates, size, tournament_size]
        return Genetic.CHOICES['selection'][self.selection_type](*args)

    def crossover(self, gens: list):
        """
        Crossovers gens to be passed in a way that was selected

        :param gens: list, candidates among witch to select.
        :return: list, child gens made from crossovering parents.
        """
        return Genetic.CHOICES['selection'][self.crossover_type](gens)

    def mutation(self, gens, mutation_rate=0.01, mutation_strength=0.05):
        """
        Makes mutations to the gens considering given limitations, strength and rate

        :param gens: list, candidates among witch to select.
        :param mutation_rate: float, probability of mutating each parameter/element(0.01 = 1%).
        :param mutation_strength: float, controls magnitude of mutations.
        :return: list, all gens after mutations.
        """
        return mut.mutate(gens, self.limitations, mutation_rate, mutation_strength)
