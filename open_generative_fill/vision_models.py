from typing import Tuple

import torch
from diffusers import AutoPipelineForInpainting
from PIL import Image
from transformers import (
    BlipForConditionalGeneration,
    BlipProcessor,
    Owlv2ForObjectDetection,
    Owlv2Processor,
    SamModel,
    SamProcessor,
)

from .utils import delete_model


def run_caption_model(model_id: str, image: Image, device: str = "cuda"):
    caption_model = BlipForConditionalGeneration.from_pretrained(model_id).to(device)
    caption_processor = BlipProcessor.from_pretrained(model_id)

    inputs = caption_processor(image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = caption_model.generate(**inputs, max_new_tokens=50)

    caption = caption_processor.decode(outputs[0], skip_special_tokens=True)
    delete_model(caption_model)
    return caption


def run_segmentaiton_pipeline(
    detection_model_id: str,
    segmentation_model_id: str,
    to_replace: str,
    image: Image,
    device: str = "cuda",
):
    processor = Owlv2Processor.from_pretrained(detection_model_id)
    od_model = Owlv2ForObjectDetection.from_pretrained(detection_model_id).to(device)
    text_queries = [to_replace]
    inputs = processor(text=text_queries, images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = od_model(**inputs)
        target_sizes = torch.tensor([image.size]).to(device)
        results = processor.post_process_object_detection(
            outputs, threshold=0.1, target_sizes=target_sizes
        )[0]

    boxes = results["boxes"].tolist()
    delete_model(od_model)

    seg_model = SamModel.from_pretrained(segmentation_model_id).to(device)
    processor = SamProcessor.from_pretrained(segmentation_model_id)

    input_boxes = [boxes]
    inputs = processor(image, input_boxes=input_boxes, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = seg_model(**inputs)

    mask = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )[0]

    mask = torch.max(mask[:, 0, ...], dim=0, keepdim=False).values
    segmentation_mask = Image.fromarray(mask.numpy())

    delete_model(seg_model)
    return segmentation_mask


def run_inpainting_pipeline(
    inpainting_model_id: str,
    image: Image,
    mask: Image,
    replaced_caption: str,
    image_size: Tuple[int, int],
    generator: torch.Generator,
    device: str = "cuda",
):
    pipeline = AutoPipelineForInpainting.from_pretrained(
        inpainting_model_id,
        torch_dtype=torch.float16,
        variant="fp16",
    ).to(device)

    prompt = replaced_caption

    negative_prompt = """lowres, bad anatomy, bad hands,
    text, error, missing fingers, extra digit, fewer digits,
    cropped, worst quality, low quality"""

    output = pipeline(
        prompt=prompt,
        image=image,
        height=image_size[0],
        width=image_size[1],
        mask_image=mask,
        negative_prompt=negative_prompt,
        guidance_scale=7.5,
        strength=1.0,
        generator=generator,
    ).images[0]
    delete_model(pipeline)
    return output
