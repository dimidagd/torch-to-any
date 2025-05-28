import numpy as np
import onnx
import onnxruntime
import torch

from torch_to_any.logger import logger
from torch_to_any.ops import ModelExporter


class ONNXExporter(ModelExporter):
    def load_model_standalone(self, output_model: str) -> onnx.ModelProto:
        return onnx.load(output_model)

    def load_model(self, output_model: str):
        ort_session = onnxruntime.InferenceSession(output_model, providers=["CPUExecutionProvider"])
        return ort_session

    def export(self, torch_model: torch.nn.Module, output_model: str) -> None:
        logger.info("Exporting model to ONNX format...")
        torch_model.eval()
        example_inputs = (super().test_sample,)

        torch.onnx.export(
            torch_model,
            example_inputs,
            f=output_model,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},  # input's 0th dim is dynamic
                "output": {0: "batch_size"},  # output's 0th dim is dynamic
            },
        )
        logger.info("ONNX model exported.")

    def validate(self, output_model: str) -> None:
        logger.info("Validating the ONNX model...")
        onnx_model = self.load_model_standalone(output_model)
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX model is valid.")
        logger.debug(f"Model graph: {onnx.helper.printable_graph(onnx_model.graph)}")

    def prepare_single_example_inputs(self, example_inputs):
        """Prepares example inputs for the model."""
        return [tensor.numpy(force=True) for tensor in example_inputs]

    def prepare_batched_example_inputs(self):
        batched = np.repeat(
            self.test_sample.numpy(force=True), repeats=self.NUM_TEST_SAMPLES, axis=0
        )
        return [batched]

    def inference(self, model, inputs) -> torch.Tensor:
        logger.debug(f"Input size: {len(inputs[0])}")
        onnx_input = {
            input_arg.name: input_val for input_arg, input_val in zip(model.get_inputs(), inputs)
        }
        onnx_outputs = model.run(None, onnx_input)[0]
        return onnx_outputs

    def test(self, torch_model: torch.nn.Module, output_model: str) -> None:
        logger.info("Testing the ONNX model...")
        example_inputs = (super().test_sample,)
        onnx_inputs = self.prepare_single_example_inputs(example_inputs)
        torch_outputs = torch_model(*example_inputs)
        onnx_outputs = self.inference(self.load_model(output_model), onnx_inputs)
        assert len(torch_outputs) == len(onnx_outputs)
        for t_out, o_out in zip(torch_outputs, onnx_outputs):
            torch.testing.assert_close(t_out, torch.tensor(o_out))

        logger.info("PyTorch and ONNX Runtime output matched!")
        logger.debug(f"Output length: {len(onnx_outputs)}")
        logger.debug(f"Sample output: {onnx_outputs}")
