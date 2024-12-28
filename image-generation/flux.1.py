"""
This script uses the FluxPipeline from the diffusers library to generate an image based on a given text prompt.

Modules:
    torch: PyTorch library for tensor computations.
    diffusers: Library for diffusion models, specifically the FluxPipeline.

Device:
    The script uses the CPU for computations.

Pipeline:
    The FluxPipeline is loaded from the pretrained model "black-forest-labs/FLUX.1-dev" with specific tokenizer settings.

Prompt:
    The text prompt describes a scene where a programmer works in a rural setting with various humorous and imaginative elements.

Image Generation:
    The pipeline generates an image based on the prompt with specified parameters:
        - height: 1024 pixels
        - width: 1024 pixels
        - guidance_scale: 3.5
        - num_inference_steps: 50
        - max_sequence_length: 512
        - generator: A CPU-based random seed generator for reproducibility

Output:
    The generated image is saved as "a1.png".
"""

import torch
from diffusers import FluxPipeline

device = torch.device("cpu")  # Use CPU instead of GPU

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16, tokenizer_kwargs={"use_fast": False})
pipe = pipe.to(device)

#prompt = "A cat holding a sign that says Programmer in the Country"
#prompt = "A programmer sits on a haystack in a rural village, using a high-tech laptop while surrounded by chickens pecking at the keyboard. Wires from the laptop run to a cow chewing on a makeshift solar panel. A goat in the background watches intently, as if learning to code."
#prompt = "A programmer in the middle of a rustic wooden barn, with a high-end gaming PC powered by a windmill outside. The screen shows lines of code, while a pig wearing glasses snoops at the monitor. A rooster perched on the desk proudly crows as if debugging."
prompt = "A programmer works in a field under an old tree, with a laptop balanced on a milk crate. Wi-Fi is provided by a satellite dish taped to a scarecrow, and sheep wander by, seemingly uninterested in the tech revolution happening next to them."

image = pipe(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=3.5,
    num_inference_steps=50,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0]

image.save("a1.png")
