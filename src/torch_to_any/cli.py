import click

from torch_to_any import scripts


@click.group()
def cli():
    pass


@cli.command()
@click.argument("output_model")
def to_onnx(output_model):
    """Convert model to ONNX format."""
    scripts.to_onnx(output_model)


@cli.command()
@click.argument("output_model")
def to_torchscript(output_model):
    """Convert model to TorchScript format."""
    scripts.to_torchscript(output_model)
