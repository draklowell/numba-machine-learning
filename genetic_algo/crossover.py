"""CROSOVER MODULE"""

import random
from copy import deepcopy
import numpy as np

def crossover_arrays(array1, array2, method):
    """Helper function to crossover two numpy arrays to produce one child array"""
    shape = array1.shape

    flat1 = array1.flatten()
    flat2 = array2.flatten()
    size = flat1.size

    if method == 'single_point':
        point = random.randint(1, size - 1)
        if random.random() < 0.5:
            new_flat = np.concatenate([flat1[:point], flat2[point:]])
        else:
            new_flat = np.concatenate([flat2[:point], flat1[point:]])
        return new_flat.reshape(shape)

    elif method == 'two_point':
        point1 = random.randint(1, size - 2)
        point2 = random.randint(point1 + 1, size - 1)

        if random.random() < 0.5:
            new_flat = np.concatenate([flat1[:point1], flat2[point1:point2], flat1[point2:]])
        else:
            new_flat = np.concatenate([flat2[:point1], flat1[point1:point2], flat2[point2:]])
        return new_flat.reshape(shape)

    elif method == 'uniform':
        # Create a random mask for uniform crossover
        mask = np.random.rand(size) > 0.5
        new_flat = np.where(mask, flat1, flat2)
        return new_flat.reshape(shape)

    return new_flat.reshape(shape)

def default_crossover(gens, method: str):
    """
    Default crossover method, which acts depending on the method
    
    Args:
        gens: List of genomes (individuals)
        method: Crossover method to use ('single_point', 'two_point', 'uniform')
    
    Returns:
        New generation after crossover(based on the method)
    """
    if len(gens) < 2:
        return gens

    offspring = []

    #копіюємо останній геном, якщо непарна кількість батьків(ми не в арабських сім'ях)
    if len(gens) % 2 != 0:
        gens.append(deepcopy(gens[-1]))
    random.shuffle(gens)

    #паруємо батьків і застосовуємо кросовер
    for i in range(0, len(gens), 2):
        parent1 = gens[i]
        parent2 = gens[i + 1]

        child = deepcopy(parent1)

        for layer_name in child:
            for param_name in child[layer_name]:
                if not hasattr(parent1[layer_name][param_name], 'shape')\
                    or parent1[layer_name][param_name].size <= 1:
                    #скалярні значення, випадковий вибір з батьків
                    if random.random() < 0.5:
                        child[layer_name][param_name] = parent1[layer_name][param_name]
                    else:
                        child[layer_name][param_name] = parent2[layer_name][param_name]
                    continue

                child[layer_name][param_name] = crossover_arrays(
                    parent1[layer_name][param_name],
                    parent2[layer_name][param_name],
                    method
                )

        offspring.append(child)

    return offspring

def single_point(gens):
    """
    Single-point crossover method
    """
    return default_crossover(gens, 'single_point')

def two_point(gens):
    """
    Two-point crossover method
    """
    return default_crossover(gens, 'two_point')

def uniform(gens):
    """
    Uniform crossover method
    """
    return default_crossover(gens, 'uniform')
