"""
This script generates audio using the StableAudioPipeline from the diffusers library.

The script performs the following steps:
1. Imports necessary libraries.
2. Loads the StableAudioPipeline model from the pretrained "stabilityai/stable-audio-open-1.0" checkpoint.
3. Sets the device to CPU.
4. Defines the prompts for audio generation.
5. Sets a seed for reproducibility.
6. Runs the audio generation process with specified parameters.
7. Saves the generated audio to a file.

Dependencies:
- torch
- soundfile
- diffusers

Usage:
- Ensure the required libraries are installed.
- Run the script to generate and save the audio file.

Parameters:
- prompt (str): The text prompt for audio generation.
- negative_prompt (str): The negative prompt to avoid certain qualities in the generated audio.
- num_inference_steps (int): The number of inference steps for the generation process.
- audio_end_in_s (float): The duration of the generated audio in seconds.
- num_waveforms_per_prompt (int): The number of waveforms to generate per prompt.
- generator (torch.Generator): The random number generator for reproducibility.

Output:
- The generated audio is saved as "hammer.wav" in the current directory.
"""

import torch
import soundfile as sf
from diffusers import StableAudioPipeline

pipe = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=torch.bfloat16)
device = torch.device("cpu")
pipe = pipe.to(device)

# define the prompts
prompt = "hammer hitting a nail"
negative_prompt = "Low quality"

# set the seed for generator
generator = torch.Generator(device).manual_seed(0)

# run the generation
audio = pipe(
    prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=200,
    audio_end_in_s=10.0,
    num_waveforms_per_prompt=3,
    generator=generator,
).audios

output = audio[0].T.float().cpu().numpy()
sf.write("hammer.wav", output, pipe.vae.sampling_rate)

