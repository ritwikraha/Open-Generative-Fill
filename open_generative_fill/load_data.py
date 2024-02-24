from typing import Tuple

import requests
from PIL import Image


def load_image(image_url: str, image_size: Tuple = (512, 512)):
    response = requests.get(image_url, stream=True)
    response.raw.decode_content = True
    raw_image = Image.open(response.raw).convert("RGB")
    image = raw_image.resize(image_size)
    return image
