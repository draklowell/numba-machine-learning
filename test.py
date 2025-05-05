import sys

import numpy as np

from genetic import (
    ChromosomePipeline,
    Crossover,
    GaussianMutation,
    GenomePipeline,
    RouletteSelection,
)
from loader import DataManager, Downloader
from nml import (
    Cast,
    CellularAutomata,
    Device,
    Flatten,
    Input,
    Linear,
    ReLU,
    Sequential,
    Softmax,
)
from project import FitnessEvaluator, GenerationHandler, Manager

print("Import successful")

sequential = Sequential(
    Input((28, 28), np.dtype("uint8")),
    CellularAutomata(
        rule_bitwidth=1,
        neighborhood="moore_1",
        iterations=80,
    ),
    CellularAutomata(
        rule_bitwidth=1,
        neighborhood="moore_1",
        iterations=80,
    ),
    Flatten(),
    Cast(np.dtype("float32")),
    Linear(768),
    ReLU(),
    Linear(10),
    Softmax(),
)


parameters = sequential.build().get_parameters()

print("Initial model built")


chromosome_pipelines = {}
for name, parameter in parameters.items():
    if name.startswith("cellular_automata"):
        chromosome_pipelines[name] = ChromosomePipeline(
            mutation=GaussianMutation(parameter, rate=0.1, strength=0.5),
            crossover=Crossover("two_point"),
        )
    elif name.startswith("linear"):
        chromosome_pipelines[name] = ChromosomePipeline(
            mutation=GaussianMutation(parameter, rate=0.1, strength=0.05),
            crossover=Crossover("two_point"),
        )
    else:
        raise ValueError(f"Unknown parameter name: {name}")


pipeline = GenomePipeline(
    selection=RouletteSelection(180),
    elitarism_selection=RouletteSelection(20),
    pipelines=chromosome_pipelines,
)

print("Genome pipeline created")

downloader = Downloader("mnist")
if not downloader.download_dataset():
    raise RuntimeError("Failed to download MNIST dataset.")

print("Dataset downloaded")

manager = Manager(
    sequential=sequential,
    fitness_evaluator=FitnessEvaluator(),
    data_manager=DataManager(
        data_path="mnist/",
        labels_path="mnist/",
        bit_width=1,
        batch_size=1024,
        process_device=Device.CPU,
        storage_device=Device.CPU,
    ),
    genome_pipeline=pipeline,
    generation_handler=GenerationHandler(
        save_path="generations/{generation}.pkl",
        save_period=10,
        log_file=sys.stdout,
        log_period=1,
    ),
    device=Device.CPU,
    population_size=100,
)

print("Manager created")

manager.run(10)

print("Manager run completed")
