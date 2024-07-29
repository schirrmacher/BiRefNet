import os
import torch
import argparse

from tqdm import tqdm
from typing import List
from PIL import Image
import numpy as np

from models.birefnet import BiRefNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference using BiRefNet model.")
    parser.add_argument(
        "--image",
        "-i",
        type=str,
        required=True,
        help="Path to the input image file or folder. If a folder is specified, all images in the folder will be processed.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="e_preds",
        help="Path to the output folder.",
    )
    parser.add_argument(
        "--model-path",
        "-m",
        type=str,
        required=True,
        help="Path to the model file.",
    )
    parser.add_argument(
        "--bg-color",
        "-bg",
        type=str,
        choices=["white", "black", "grey"],
        default="white",
        help="Background color for the output images. Choices are 'white', 'black', or 'grey'. Default is 'white'.",
    )
    parser.add_argument(
        "--output-mode",
        "-om",
        type=str,
        choices=["compare", "processed", "alpha"],
        default="processed",
        help="Output mode: 'compare' to save original and processed images side by side, 'processed' to save only the processed image, or 'alpha' to save only the alpha channel as a black and white image.",
    )
    return parser.parse_args()


def load_images_from_folder(folder: str) -> List[str]:
    image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff")
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(image_extensions)
    ]


def check_state_dict(state_dict, unwanted_prefix="_orig_mod."):
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    return state_dict


def apply_mask_to_image(
    image_path: str, mask_tensor: torch.Tensor, bg_color: str, output_mode: str
) -> Image:
    mask = mask_tensor.squeeze().cpu().numpy()
    mask = (mask * 255).astype(np.uint8)
    mask_image = Image.fromarray(mask).convert("L")

    original_image = Image.open(image_path).convert("RGB")
    mask_image = mask_image.resize(original_image.size, Image.BILINEAR)

    # Determine the background color
    if bg_color == "white":
        bg_rgba = (255, 255, 255)
    elif bg_color == "black":
        bg_rgba = (0, 0, 0)
    elif bg_color == "grey":
        bg_rgba = (128, 128, 128)
    else:
        bg_rgba = (0, 0, 0)  # default to black if unknown

    # Create background image
    background = Image.new("RGB", original_image.size, bg_rgba)

    # Composite the original image with the background using the mask
    result_image = Image.composite(original_image, background, mask_image)

    if output_mode == "compare":
        combined_width = original_image.width + result_image.width
        combined_image = Image.new("RGB", (combined_width, original_image.height))
        combined_image.paste(original_image, (0, 0))
        combined_image.paste(result_image, (original_image.width, 0))
        return combined_image
    elif output_mode == "alpha":
        return mask_image
    else:
        return result_image


def inference(
    model: torch.nn.Module,
    image_paths: List[str],
    pred_root: str,
    method: str,
    device: torch.device,
    bg_color: str,
    output_mode: str,
) -> None:
    model_training = model.training
    if model_training:
        model.eval()

    os.makedirs(pred_root, exist_ok=True)

    fixed_size = (1024, 1024)  # Set a fixed size for the input images

    try:
        for image_path in tqdm(image_paths, total=len(image_paths)):
            output_path = os.path.join(pred_root, os.path.basename(image_path))
            if os.path.exists(output_path):
                print(
                    f"Skipping {image_path} as it already exists in the output folder."
                )
                continue

            image = Image.open(image_path).convert("RGB")
            image = image.resize(fixed_size, Image.BILINEAR)
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).to(device)

            with torch.no_grad():
                scaled_preds = model(image)[-1].sigmoid()

            res = torch.nn.functional.interpolate(
                scaled_preds, size=fixed_size, mode="bilinear", align_corners=True
            )
            result_image = apply_mask_to_image(image_path, res, bg_color, output_mode)
            result_image.save(output_path)

    except KeyboardInterrupt:
        print("Process interrupted by user. Exiting gracefully.")
    finally:
        if model_training:
            model.train()


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Testing with model {args.model_path}")

    model = BiRefNet(bb_pretrained=False)

    state_dict = torch.load(args.model_path, map_location="cpu")
    state_dict = check_state_dict(state_dict)
    model.load_state_dict(state_dict)
    model = model.to(device)

    # Determine if the input is a single image or a folder
    if os.path.isdir(args.image):
        image_paths = load_images_from_folder(args.image)
    else:
        image_paths = [args.image]

    print(f">>>> Processing images in {args.image}...")
    print(f"\tInferencing {args.model_path}...")
    inference(
        model=model,
        image_paths=image_paths,
        pred_root=args.output,
        method=os.path.basename(args.model_path).rstrip(".pth"),
        device=device,
        bg_color=args.bg_color,
        output_mode=args.output_mode,
    )


if __name__ == "__main__":
    args = parse_args()
    try:
        main(args)
    except KeyboardInterrupt:
        print("Process interrupted by user. Exiting gracefully.")
