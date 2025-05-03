from genetic.crossover import Crossover
from genetic.mutation import Mutation
from nml import Device, Tensor, copy_to

try:
    from numba import cuda
except ImportError:
    cuda = None


class ChromosomePipeline:
    """
    A class that represents a pipeline for processing chromosome pairs.
    It applies a crossover operation followed by a mutation operation.
    """

    crossover: Crossover
    mutation: Mutation
    process_device: Device
    output_device: Device

    def __init__(
        self,
        crossover: Crossover,
        mutation: Mutation,
        process_device: Device = Device.CPU,
        output_device: Device = Device.CPU,
    ):
        self.crossover = crossover
        self.mutation = mutation
        self.process_device = process_device
        self.output_device = output_device

    def __call__(self, pairs: list[tuple[Tensor, Tensor]]) -> list[Tensor]:
        """
        Apply the pipeline to a list of pairs of tensors (chromosome pairs).
        """
        if self.process_device == Device.GPU and cuda is not None:
            streams = []
            result = []
            for pair in pairs:
                stream = cuda.stream()
                streams.append(stream)
                ctx = {"cuda.stream": stream}

                pair = tuple(
                    copy_to(parent, self.process_device, ctx) for parent in pair
                )
                offspring = self.crossover(pair, ctx)
                mutated_offspring = self.mutation(offspring, ctx)

                result.append(copy_to(mutated_offspring, self.output_device, ctx))

            for stream in streams:
                stream.synchronize()

            return result

        if self.process_device == Device.CPU:
            result = []
            for pair in pairs:
                pair = tuple(
                    copy_to(parent, self.process_device, {}) for parent in pair
                )
                offspring = self.crossover(pair, {})
                mutated_offspring = self.mutation(offspring, {})
                result.append(copy_to(mutated_offspring, self.output_device, {}))

            return result

        raise NotImplementedError(f"Device {self.process_device} is not supported")
