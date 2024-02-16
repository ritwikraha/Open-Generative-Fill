# USAGE:
# python inference.py --image_url "https://i.imgur.com/4ujXoav.jpeg" --edit_prompt "change the bottle to a firecracker"
import argparse

import torch

from open_generative_fill import config
from open_generative_fill.lm_models import run_lm_model
from open_generative_fill.load_data import load_image
from open_generative_fill.vision_models import (
    run_caption_model,
    run_inpainting_pipeline,
    run_segmentaiton_pipeline,
)


def load_arguments(parser):
    parser = argparse.ArgumentParser(
        description="Inference for the Open Generative Fill"
    )
    parser.add_argument(
        "--image_url", type=str, help="The image url to use for inference"
    )
    parser.add_argument(
        "--edit_prompt", type=str, help="The edit prompt to use for inference"
    )

    return parser.parse_args()


if __name__ == "__main__":
    # Get the arguments from the command line
    parser = argparse.ArgumentParser(
        description="Inference for the Open Generative Fill"
    )
    arguments = load_arguments(parser)

    # Set the device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the image from the url and get the text prompt
    image_url = arguments.image_url
    image = load_image(image_url=image_url, image_size=config.IMAGE_SIZE)
    edit_prompt = arguments.edit_prompt

    # Image captioning
    caption = run_caption_model(
        model_id=config.CAPTION_MODEL_ID, image=image, device=device
    )

    # Language model
    to_replace, replaced_caption = run_lm_model(
        model_id=config.LANGUAGE_MODEL_ID,
        caption=caption,
        edit_prompt=edit_prompt,
        device=device,
    )

    # Segmentation pipeline
    segmentation_mask = run_segmentaiton_pipeline(
        detection_model_id=config.DETECTION_MODEL_ID,
        segmentation_model_id=config.SEGMENTATION_MODEL_ID,
        to_replace=to_replace,
        image=image,
        device=device,
    )

    # Inpainting pipeline
    output = run_inpainting_pipeline(
        inpainting_model_id=config.INPAINTING_MODEL_ID,
        image=image,
        mask=segmentation_mask,
        replaced_caption=replaced_caption,
        image_size=config.IMAGE_SIZE,
        generator=torch.Generator().manual_seed(17),
        device=device,
    )
