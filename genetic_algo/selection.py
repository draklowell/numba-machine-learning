"""
Provides functions to 3 different methods of selection: tournament, roulette, rank
"""
from random import sample, uniform

def tournament(candidates:list, size:int, tournament_size:int=2) -> list:
    """
    Randomly selects tournament_size number of candidates and chooses the bes among them.
    Repeates so untile gets resulting list with given size

    :param candidates: list, candidates among witch to select.
    :param size: int, number of best gens to be selected.
    :param tournament_size: int, number of gens among witch to select during one tournament
    :return: list, gens selected among given.
    """
    result = []
    while size:
        tourn = sample(candidates, k=tournament_size)
        result.append(max(tourn, key=lambda x:x[1])[0])
        size -=1
    return result

def roulette(candidates:list, size:int) -> list:
    """
    Randomly selects given number of individuals where fitness is 
    a possibility of each individual to be selected.

    :param candidates: list, candidates among witch to select.
    :param size: int, number of best gens to be selected.
    :return: list, gens selected among given.
    """
    result = []
    total_fitness = 0

    for candidate in candidates:
        total_fitness += candidate[1]

    while size:
        pick = uniform(0, total_fitness)
        current = 0
        for candidate, fitness in candidates:
            current += fitness
            if current >= pick:
                result.append(candidate)
                break
        size -= 1

    return result

def rank(candidates:list, size:int):
    """
    Randomly selects given number of individuals where rank compered to other genomes
    is a possibility of each individual to be selected.

    For example for candidates [(G1, 31), (G2, 28), (G3, 98), (G4, 42)] ranking will be:
    4-G3, 3-G4, 2-G1, 1-G2
    So the highest possibility to be selected will be in G3 (4/(4+3+2+1))

    :param candidates: list, candidates among witch to select.
    :param size: int, number of best gens to be selected.
    :return: list, gens selected among given.
    """
    result = []
    sorted_candidates = sorted(candidates, key=lambda x: x[1])

    ranks = []
    ranks_sum = 0
    for x in range(len(sorted_candidates), 0, -1):
        ranks.append(x)
        ranks_sum += x

    while size:
        pick = uniform(0, ranks_sum)
        current = 0
        for candidate, rank_ in zip([c[0] for c in sorted_candidates], ranks):
            current += rank_
            if current >= pick:
                result.append(candidate)
                break
        size -= 1

    return result
