from torch_to_any.modeling.models import (
    ImageClassifierModel,
    resnet,
    resnet_expected_sample,
    classifier_expected_sample,
)


def test_image_classifier_model_initialization():
    """Test that the ImageClassifierModel initializes correctly."""
    model = ImageClassifierModel()
    assert isinstance(model, ImageClassifierModel)


def test_image_classifier_model_forward_pass():
    """Test the forward pass of the ImageClassifierModel."""
    model = ImageClassifierModel()
    model.eval()  # Set the model to evaluation mode
    input_tensor = classifier_expected_sample  # Example input
    output = model(input_tensor)
    assert output.shape == (
        classifier_expected_sample.shape[0],
        model.fc3.out_features,
    )  # Expected output shape


def test_resnet_initialization():
    """Test that the resnet function loads a pretrained ResNet-50 model."""
    model = resnet()
    assert model.__class__.__name__ == "ResNet"
    assert model.fc.out_features == 1000  # ResNet-50 default number of classes


def test_resnet_forward_pass():
    """Test the forward pass of the ResNet-50 model."""
    model = resnet()
    model.eval()  # Set the model to evaluation mode
    input_tensor = resnet_expected_sample  # Use the predefined sample input
    output = model(input_tensor)
    assert output.shape == (
        resnet_expected_sample.shape[0],
        model.fc.out_features,
    )  # Expected output shape for ResNet-50
