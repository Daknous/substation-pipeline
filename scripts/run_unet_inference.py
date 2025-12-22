import argparse
import sys
from pathlib import Path
import torch

# Make repo root + "src" importable (so both src.* and helpers.* work)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from unet_infer.load_model import load_unet_model, build_preprocess
from unet_infer.transformer_pipeline import BatchTransformerProcessor


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--model_pth", required=True)

    ap.add_argument("--pattern", default="*.png")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])

    ap.add_argument("--encoder_name", default="resnet34")
    ap.add_argument("--encoder_weights", default=None)  # "imagenet" or None
    ap.add_argument("--in_channels", type=int, default=3)
    ap.add_argument("--classes", type=int, default=1)

    ap.add_argument("--allow_no_scale", action="store_true")
    ap.add_argument("--scale_manifest_csv", default=None)
    ap.add_argument("--metadata_json", default=None)

    ap.add_argument("--save_interval", type=int, default=100)
    ap.add_argument("--debug_bridge", action="store_true")
    ap.add_argument("--skip_existing", action="store_true",
                    help="Skip images that already have overlay/mask/CSV entries")
    return ap.parse_args()


def main():
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model = load_unet_model(
        checkpoint_path=args.model_pth,
        device=device,
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        in_channels=args.in_channels,
        classes=args.classes,
    )
    preprocess = build_preprocess(args.encoder_name, args.encoder_weights)

    proc = BatchTransformerProcessor(
        output_dir=args.output_dir,
        model=model,
        preprocess=preprocess,
        device=device,
        allow_no_scale=args.allow_no_scale,
        debug_bridge=args.debug_bridge,
        skip_existing=args.skip_existing,
    )

    proc.process_directory(
        image_dir=args.images_dir,
        pattern=args.pattern,
        metadata_json=args.metadata_json,
        save_interval=args.save_interval,
        scale_manifest_csv=args.scale_manifest_csv,
    )


if __name__ == "__main__":
    main()
    