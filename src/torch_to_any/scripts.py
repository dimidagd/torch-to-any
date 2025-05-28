"""Scripts of the project."""

# %% IMPORTS

import sys
from torch_to_any.modeling import ImageClassifierModel as Model
# %% FUNCTIONS


def main(argv: list[str] | None = None) -> int:
    """Run the main script function."""
    args = argv or sys.argv[1:]
    print("Args:", args)
    return 0


def to_onnx(output_model: str) -> None:
    """Convert a PyTorch model to ONNX format."""
    torch_model = Model()
    from onnx_ops import onnx_export_pipeline
    onnx_export_pipeline(torch_model, output_model)
