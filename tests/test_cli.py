import pytest
from click.testing import CliRunner
from torch_to_any import cli


@pytest.fixture
def runner():
    """Fixture for Click CLI runner."""
    return CliRunner()


@pytest.mark.parametrize("output_path", ["/tmp/model.onnx", "/var/tmp/model.onnx", "./model.onnx"])
def test_to_onnx_command(mocker, runner, output_path):
    """Test the to_onnx CLI command."""
    mock_to_onnx = mocker.patch("torch_to_any.scripts.to_onnx")  # Mock the to_onnx function
    result = runner.invoke(cli, ["to-onnx", "/tmp/model.onnx"])
    # Assertions
    assert result.exit_code == 0
    mock_to_onnx.assert_called_once_with("/tmp/model.onnx")
    assert "Model exported to ONNX format at /tmp/model.onnx" in result.output


@pytest.mark.parametrize("output_path", ["/tmp/model.pt", "/var/tmp/model.pt", "./model.pt"])
def test_to_torchscript_command(mocker, runner, output_path):
    """Test the to_torchscript CLI command."""
    mock_to_torchscript = mocker.patch(
        "torch_to_any.scripts.to_torchscript"
    )  # Mock the to_torchscript function
    result = runner.invoke(cli, ["to-torchscript", "/tmp/model.pt"])
    # Assertions
    assert result.exit_code == 0
    mock_to_torchscript.assert_called_once_with("/tmp/model.pt")
    assert "Model exported to TorchScript format at /tmp/model.pt" in result.output
