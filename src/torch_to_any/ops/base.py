import time
from abc import ABC, abstractmethod

import torch
import onnx
import onnxruntime
from torch_to_any.logger import logger
from torch_to_any.modeling import resnet_expected_sample


class ModelExporter(ABC):
    NUM_TEST_SAMPLES = 100
    test_sample = resnet_expected_sample

    @abstractmethod
    def export(self, torch_model: torch.nn.Module, output_model: str) -> None:
        """Exports the given PyTorch model to the desired format."""
        pass

    @abstractmethod
    def validate(self, output_model: str) -> None:
        """Validates the exported model."""
        pass

    @abstractmethod
    def load_model(
        self, output_model: str
    ) -> torch.nn.Module | torch.jit.ScriptModule | onnx.ModelProto | onnxruntime.InferenceSession:
        """Loads the exported model and returns it."""
        pass

    @abstractmethod
    def inference(self, model, example_inputs: tuple[torch.Tensor, ...]):
        """Performs inference on the model with the given inputs."""
        pass

    @abstractmethod
    def prepare_single_example_inputs(self, example_inputs: tuple[torch.Tensor]):
        """Prepares example inputs for the model."""
        pass

    def get_example_inputs(self) -> tuple[torch.Tensor, ...]:
        """Returns a single example input for testing."""
        return (self.test_sample,)

    @abstractmethod
    def prepare_batched_example_inputs(self):
        pass

    def measure_throughput(self, output_model):
        model = self.load_model(output_model)
        example_inputs = self.prepare_batched_example_inputs()
        start_time = time.time()
        self.inference(model, example_inputs)
        elapsed_time = time.time() - start_time
        logger.info(f"Forward pass took {elapsed_time:.6f} seconds.")

    @abstractmethod
    def test(self, torch_model: torch.nn.Module, output_model: str) -> None:
        """Tests the exported model by comparing inference outputs."""
        pass

    def export_pipeline(self, torch_model: torch.nn.Module, output_model: str) -> None:
        """Full pipeline: export, validate, and test."""
        from torch_to_any.logger import logger

        logger.info(
            f"Exporting model to {self.__class__.__name__.replace('Exporter', '')} format..."
        )
        self.export(torch_model, output_model)

        logger.info("Validating the exported model...")
        self.validate(output_model)

        logger.info("Testing the exported model...")
        self.test(torch_model, output_model)

        logger.info("Export and validation completed successfully.")

        self.measure_throughput(output_model)
