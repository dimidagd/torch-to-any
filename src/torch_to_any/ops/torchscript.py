import torch

from torch_to_any.logger import logger
from torch_to_any.ops import ModelExporter


class TorchScriptExporter(ModelExporter):
    def export(self, torch_model: torch.nn.Module, output_model: str) -> None:
        """Export the PyTorch model to TorchScript format."""
        logger.info("Exporting model to TorchScript format...")
        torch_model.eval()  # Ensure the model is in evaluation mode
        scripted_model = torch.jit.script(torch_model)
        scripted_model.save(output_model)
        logger.info(f"TorchScript model exported to {output_model}.")

    def load_model(self, output_model: str) -> torch.jit.ScriptModule:
        """Load the TorchScript model from the specified file."""
        logger.info(f"Loading TorchScript model from {output_model}...")
        scripted_model = torch.jit.load(output_model)
        scripted_model.eval()
        return scripted_model

    def prepare_single_example_inputs(
        self, example_inputs: tuple[torch.Tensor]
    ) -> tuple[torch.Tensor, ...]:
        """Prepares example inputs for the model."""
        return example_inputs

    def prepare_batched_example_inputs(self):
        return (torch.cat([super().test_sample for _ in range(super().NUM_TEST_SAMPLES)]),)

    def validate(self, output_model: str) -> None:
        """Validate the exported TorchScript model."""
        logger.info("Validating the TorchScript model...")
        try:
            self.load_model(output_model)
            logger.info("TorchScript model is valid and successfully loaded.")
        except Exception as e:
            logger.error(f"Failed to validate TorchScript model: {e}")
            raise

    def inference(self, model, example_inputs):
        return model(*example_inputs)

    def test(self, torch_model: torch.nn.Module, output_model: str) -> None:
        """Test the exported TorchScript model."""
        logger.info("Testing the TorchScript model...")
        example_inputs = (super().test_sample,)
        torch_model.eval()
        scripted_model = self.load_model(output_model)
        torchscript_inputs = self.prepare_single_example_inputs(example_inputs)
        torch_outputs = torch_model(*example_inputs)
        scripted_outputs = scripted_model(*torchscript_inputs)

        assert len(torch_outputs) == len(scripted_outputs)
        for t_out, s_out in zip(torch_outputs, scripted_outputs):
            torch.testing.assert_close(t_out, s_out)

        logger.info("PyTorch and TorchScript outputs matched!")
        logger.debug(f"Sample output: {scripted_outputs}")
