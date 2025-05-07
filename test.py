import os

import numpy as np

from genetic import (
    BestSelection,
    ChromosomePipeline,
    Crossover,
    GaussianMutation,
    GenomePipeline,
    RouletteSelection,
)
from handlers import TableHandler, PrintHandler, SaveHandler
from loader import Downloader, SklearnBalancedDataLoader
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
from project import FitnessEvaluator, Manager

print("Import successful")

sequential = Sequential(
    Input((8, 8), np.dtype("uint8")),
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
    selection=RouletteSelection(14),
    elitarism_selection=BestSelection(3),
    pipelines=chromosome_pipelines,
)

print("Genome pipeline created")

downloader = Downloader("datasets/mnist")
if not downloader.download_dataset():
    raise RuntimeError("Failed to download MNIST dataset.")

downloader.create_numpy_dataset(
    "datasets/mnist/train-images-idx3-ubyte.npy",
)

print("Dataset downloaded")


sklear_manager = SklearnBalancedDataLoader(
    batch_size=10,
    process_device=Device.CPU,
    storage_device=Device.CPU,
    random_state=42,
)


print("Data manager created")

os.makedirs("generations", exist_ok=True)
manager = Manager(
    sequential=sequential,
    fitness_evaluator=FitnessEvaluator(),
    data_manager=sklear_manager,
    genome_pipeline=pipeline,
    handlers=[
        TableHandler(
            log_file=open("log.csv", "w"),
            log_period=1,
            profile_file=open("profile.csv", "w"),
            profile_period=1,
        ),
        PrintHandler(period=1),
        SaveHandler(
            path="generations/{generation}.pkl",
            period=10,
        ),
    ],
    device=Device.CPU,
    population_size=10,
)

print("Manager created")

manager.run(10)

print("Manager run completed")
