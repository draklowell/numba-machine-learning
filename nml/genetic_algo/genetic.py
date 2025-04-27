"""
assemblance of all the functions of genetic algo
"""
import crossover as cross
import selection as select

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
        You may use different genetic algoritms

        TODO
        :param limitations: list(?), limitations for mutations
        :param selection_type: str, may be one of (tournament, roulette, rank)
        :param crossover_type: str, may be one of (single_point, two_point, uniform)
        """
        if selection_type not in Genetic.CHOICES['selection']:
            raise ValueError(f"Bad input for selection_type. Must be one of: {tuple(x for x in Genetic.CHOICES['selection'])}")
        if crossover_type not in Genetic.CHOICES['crossover']:
            raise ValueError(f"Bad input for selection_type. Must be one of: {tuple(x for x in Genetic.CHOICES['crossover'])}")
        self.limitations = limitations
        self.selection_type = selection_type
        self.crossover_type = crossover_type

    def selection(self, gens, fitness):
        """
        Selects gens to be passed based on fitness and in a way that was selected
        """
        return Genetic.CHOICES['selection'][self.selection_type](gens, fitness)

    def crossover(self, gens):
        """
        Crossovers gens to be passed in a way that was selected
        """
        return Genetic.CHOICES['selection'][self.crossover_type](gens)

    def mutation(self, gens):
        """
        makes mutations to the gens considering given limitations
        """
        pass