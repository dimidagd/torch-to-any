from logger import logger
import onnxruntime
import onnx
import torch


def onnx_export_pipeline(torch_model, output_model: str) -> None:
    """Export the PyTorch model to ONNX format and validate it."""
    logger.info("Exporting model to ONNX format...")
    save_as_onnx(torch_model, output_model)
    
    logger.info("Validating the ONNX model...")
    validate_onnx(output_model)
    
    logger.info("Testing the ONNX model...")
    test_onnx(torch_model, output_model)
    
    logger.info("ONNX export and validation completed successfully.")
def save_as_onnx(torch_model, output_model: str) -> None:
    # Create example inputs for exporting the model. The inputs should be a tuple of tensors.
    example_inputs = (torch.randn(1, 1, 32, 32),)
    onnx_program = torch.onnx.export(torch_model, example_inputs, dynamo=True)
    onnx_program.optimize()
    onnx_program.save(output_model)
    

def validate_onnx(model_path: str) -> None:
    """Validate the ONNX model by checking its structure."""
    
    # Load the ONNX model
    onnx_model = onnx.load(model_path)

    # Check the model's structure
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX model is valid.")

    # Print the model's graph
    logger.info(f"Model graph: {onnx.helper.printable_graph(onnx_model.graph)}")

def test_onnx(torch_model, model_path: str) -> None:
    """Test the ONNX model by running inference and comparing with PyTorch."""
    example_inputs = (torch.randn(1, 1, 32, 32),)
    onnx_inputs = [tensor.numpy(force=True) for tensor in example_inputs]
    logger.info(f"Input size: {onnx_inputs[0].shape}")

    ort_session = onnxruntime.InferenceSession(
        model_path, providers=["CPUExecutionProvider"]
    )

    onnxruntime_input = {input_arg.name: input_value for input_arg, input_value in zip(ort_session.get_inputs(), onnx_inputs)}

    # ONNX Runtime returns a list of outputs
    onnxruntime_outputs = ort_session.run(None, onnxruntime_input)[0]
    # Lets compare with torch run
    torch_outputs = torch_model(*example_inputs)

    assert len(torch_outputs) == len(onnxruntime_outputs)
    for torch_output, onnxruntime_output in zip(torch_outputs, onnxruntime_outputs):
        torch.testing.assert_close(torch_output, torch.tensor(onnxruntime_output))

    logger.info("PyTorch and ONNX Runtime output matched!")
    logger.info(f"Output length: {len(onnxruntime_outputs)}")
    logger.info(f"Sample output: {onnxruntime_outputs}")
    