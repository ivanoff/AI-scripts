"""
This script uses the Stable Diffusion InstructPix2Pix model to modify an input image based on a given prompt.

Modules:
    - PIL: Python Imaging Library for image processing.
    - requests: For making HTTP requests to download images.
    - torch: PyTorch library for tensor computations.
    - diffusers: For using the Stable Diffusion InstructPix2Pix model and scheduler.

Functions:
    - download_image(url): Downloads an image from a given URL, processes it, and returns it as an RGB image.
    - download_local_image(url): Loads a local image from a given file path, processes it, and returns it as an RGB image.

Variables:
    - model_id: The identifier for the pre-trained InstructPix2Pix model.
    - pipe: The pipeline object for the InstructPix2Pix model.
    - device: The device to run the model on (CPU in this case).
    - prompt: The text prompt to instruct the model on how to modify the image.
    - image: The input image to be modified.
    - images: The list of output images generated by the model.

Usage:
    - The script loads a local image, modifies it based on the given prompt using the InstructPix2Pix model, and saves the output image.
"""

import PIL
import requests
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

model_id = "timbrooks/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16, safety_checker=None)
device = torch.device("cpu")
pipe.to(device)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

#url = "https://raw.githubusercontent.com/timothybrooks/instruct-pix2pix/main/imgs/example.jpg"
def download_image(url):
    image = PIL.Image.open(requests.get(url, stream=True).raw)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image
#image = download_image(url)

def download_local_image(url):
    image = PIL.Image.open(url)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image

prompt = "turn she happy"

image = download_local_image("./111.jpg")
images = pipe(prompt, image=image, num_inference_steps=40, image_guidance_scale=1.9).images
images[0].save("i931.png")