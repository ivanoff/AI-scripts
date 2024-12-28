"""
This script utilizes the Stable Diffusion 3.5 model to generate images based on text prompts.

# tested on 20 Gb GPU
# using abt. 9627MiB GPU
# abt. 2.2 sec/iteration

Dependencies:
- diffusers
- torch
- os

The script performs the following steps:
1. Imports necessary modules and configurations from the diffusers library and torch.
2. Defines the model ID for the Stable Diffusion 3.5 model.
3. Configures the model to use 4-bit quantization with specific settings.
4. Loads the pre-trained model with the specified quantization configuration.
5. Initializes the Stable Diffusion 3 Pipeline with the loaded model.
6. Enables model CPU offload to manage GPU memory usage.
7. Defines a function `generate_image` that:
    - Generates a unique filename for the output image.
    - Uses the pipeline to generate an image based on the provided prompt.
    - Saves the generated image to the unique filename.
    - Returns the filename of the saved image.
8. Calls the `generate_image` function with three different prompts to generate and save images.

Usage:
- Ensure you have a GPU with at least 20 GB of memory.
- Install the required libraries: diffusers, torch.
- Run the script to generate images based on the provided prompts.
"""

from diffusers import BitsAndBytesConfig, SD3Transformer2DModel
from diffusers import StableDiffusion3Pipeline
import torch

model_id = "stabilityai/stable-diffusion-3.5-large"

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model_nf4 = SD3Transformer2DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    quantization_config=nf4_config,
    torch_dtype=torch.bfloat16
)

pipeline = StableDiffusion3Pipeline.from_pretrained(
    model_id, 
    transformer=model_nf4,
    torch_dtype=torch.bfloat16
)
pipeline.enable_model_cpu_offload()

import os
def generate_image(prompt):
    counter = 1
    filename = f"ss{counter}-2.png"
    while os.path.exists(filename):
        counter += 1
        filename = f"ss{counter}-2.png"

    image = pipeline(
        prompt=prompt,
        num_inference_steps=60,
        guidance_scale=10,
        max_sequence_length=512,
        height=720,
        width=1920,
    ).images[0]
    image.save(filename)
    return filename

generate_image("A massive cargo ship sailing on a calm sea during sunrise, with vivid orange and pink hues in the sky reflecting on the water. The ship's deck is loaded with colorful shipping containers, and a faint mist surrounds the horizon.")
generate_image("A gigantic oil tanker docked at a busy port at night, illuminated by bright industrial lights and surrounded by cranes. The dark water reflects the glowing lights, creating a dramatic scene under a clear night sky.")
generate_image("Several big fishing ships floating on a serene river at dawn, surrounded by lush green hills and a thin layer of fog. Fishermen are visible on the boats with nets and gear, and the water ripples softly around them.")
