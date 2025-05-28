"""Scripts of the project."""

# %% IMPORTS

import sys

from torch_to_any.cli import cli
from torch_to_any.modeling import resnet as Model
from torch_to_any.ops.onnx import ONNXExporter
from torch_to_any.ops.torchscript import TorchScriptExporter

# %% FUNCTIONS


def main(argv: list[str] | None = None) -> int:
    """Run the main script function."""
    args = argv or sys.argv[1:]
    print("Args:", args)
    cli()
    return 0


def to_onnx(output_model: str) -> None:
    """Convert a PyTorch model to ONNX format."""
    torch_model = Model()
    torch_model.eval()  # Set the model to evaluation mode

    exporter = ONNXExporter()
    exporter.export_pipeline(torch_model, output_model)


def to_torchscript(output_model: str) -> None:
    """Convert a PyTorch model to TorchScript format."""
    torch_model = Model()
    torch_model.eval()
    exporter = TorchScriptExporter()
    exporter.export_pipeline(torch_model, output_model)
