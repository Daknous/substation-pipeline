import torch
import segmentation_models_pytorch as smp


def load_unet_model(
    checkpoint_path: str,
    device: torch.device,
    encoder_name: str = "resnet34",
    encoder_weights: str = None,
    in_channels: int = 3,
    classes: int = 1,
):
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
    )

    state = torch.load(checkpoint_path, map_location=device)
    state_dict = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state

    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    return model


def build_preprocess(encoder_name: str, encoder_weights: str = None):
    """
    Returns a callable: np.uint8 HWC -> np.float32 HWC (preprocessed)
    """
    fn = smp.encoders.get_preprocessing_fn(encoder_name, encoder_weights)
    return lambda img: fn(img)
