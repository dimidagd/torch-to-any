import time
from abc import ABC, abstractmethod

import torch

from torch_to_any.logger import logger
from torch_to_any.modeling import expected_sample


class ModelExporter(ABC):
    NUM_TEST_SAMPLES = 100
    test_sample = expected_sample

    @abstractmethod
    def export(self, torch_model: torch.nn.Module, output_model: str) -> None:
        """Exports the given PyTorch model to the desired format."""
        pass

    @abstractmethod
    def validate(self, output_model: str) -> None:
        """Validates the exported model."""
        pass

    @abstractmethod
    def load_model(self, output_model: str) -> None:
        """Measures throughput of model."""
        pass

    @abstractmethod
    def inference(self, model, example_inputs: tuple[torch.Tensor, ...]):
        """Performs inference on the model with the given inputs."""
        pass

    @abstractmethod
    def prepare_single_example_inputs(self):
        """Prepares example inputs for the model."""
        pass

    @abstractmethod
    def prepare_batched_example_inputs(self):
        pass

    def measure_throughput(self, output_model):
        model = self.load_model(output_model)
        example_inputs = self.prepare_batched_example_inputs()
        start_time = time.time()
        outputs = self.inference(model, example_inputs)
        elapsed_time = time.time() - start_time
        logger.info(f"Forward pass took {elapsed_time:.6f} seconds.")

    @abstractmethod
    def test(self, torch_model: torch.nn.Module, output_model: str) -> None:
        """Tests the exported model by comparing inference outputs."""
        pass

    def export_pipeline(self, torch_model: torch.nn.Module, output_model: str) -> None:
        """Full pipeline: export, validate, and test."""
        from torch_to_any.logger import logger

        logger.info(f"Exporting model to {self.__class__.__name__.replace('Exporter','')} format...")
        self.export(torch_model, output_model)

        logger.info("Validating the exported model...")
        self.validate(output_model)

        logger.info("Testing the exported model...")
        self.test(torch_model, output_model)

        logger.info("Export and validation completed successfully.")

        self.measure_throughput(output_model)
