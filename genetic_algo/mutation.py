"""MUTATION MODULE"""

import random
from copy import deepcopy
import numpy as np

def mutate_binary_array(array, mutation_rate=0.01):
    """
    Mutate binary values by flipping bits with probability mutation_rate
    """
    mutated = array.copy()
    mask = np.random.random(array.shape) < mutation_rate
    mutated[mask] = 1 - mutated[mask]
    return mutated

def mutate_scalar(value, low, high, mutation_strength=0.05):
    """
    Mutate a scalar value within given bounds
    """
    range_size = high - low
    delta = np.random.normal(0, mutation_strength * range_size)
    new_value = value + delta
    return np.clip(new_value, low, high)

def mutate_tensor(tensor, mutation_rate=0.01, mutation_strength=0.05):
    """
    Mutate tensor values using Gaussian noise
    """
    mutated = tensor.copy()

    mask = np.random.random(tensor.shape) < mutation_rate
    std = mutation_strength * (np.std(tensor) if  np.std(tensor) > 0 else 0.1)

    noise = np.random.normal(0, std, tensor.shape)
    mutated[mask] += noise[mask]

    return mutated

def mutate(gens, limitations, mutation_rate=0.01, mutation_strength=0.05):
    """
    Apply mutations to genomes based on their types and limitations
    
    Args:
        gens: List of genomes to mutate
        limitations: Dictionary of parameter limitations
        mutation_rate: Probability of mutating each parameter/element
        mutation_strength: Controls magnitude of mutations
    
    Returns:
        List of mutated genomes
    """
    mutated_gens = []

    for genome in gens:
        mutated_genome = deepcopy(genome)

        for layer_name in mutated_genome:
            for param_name in mutated_genome[layer_name]:
                param_limits = limitations.get(layer_name, {}).get(param_name, None)
                if param_limits is None:
                    continue

                param_value = mutated_genome[layer_name][param_name]

                # Binary parameters(cellular automatas)
                if param_limits.dtype == np.dtype('uint8') and\
                    np.all(np.logical_or(param_value == 0, param_value == 1)):
                    mutated_genome[layer_name][param_name] =\
                    mutate_binary_array(param_value, mutation_rate)

                # Scalar parameters
                elif not hasattr(param_value, 'shape') or param_value.size <= 1:
                    low = param_limits.low if hasattr(param_limits, 'low') else None
                    high = param_limits.high if hasattr(param_limits, 'high') else None

                    if random.random() < mutation_rate and low is not None and high is not None:
                        mutated_genome[layer_name][param_name] = mutate_scalar(
                            param_value, low, high, mutation_strength
                        )

                else:
                    mutated_genome[layer_name][param_name] = mutate_tensor(
                        param_value, mutation_rate, mutation_strength
                    )

        mutated_gens.append(mutated_genome)

    return mutated_gens
