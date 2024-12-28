"""
This script uses the Hugging Face Transformers library to generate a music track based on a text prompt.

Imports:
    from transformers import pipeline: Imports the pipeline function from the transformers library.
    import scipy: Imports the scipy library for handling audio file operations.

Variables:
    synthesiser: Initializes a text-to-audio pipeline using the "facebook/musicgen-small" model.
    music: Generates a music track based on the provided text prompt with specific parameters.

Functionality:
    - The script initializes a text-to-audio pipeline using the "facebook/musicgen-small" model.
    - It generates a music track based on a detailed text prompt describing the desired characteristics of the track.
    - The generated music is saved as a WAV file using the scipy library.

Output:
    The generated music is saved as "musicgen_out10.wav" with the specified sampling rate and audio data.
"""

from transformers import pipeline
import scipy

synthesiser = pipeline("text-to-audio", "facebook/musicgen-small")

music = synthesiser("Create a track in the soft rock genre, fast-paced and dynamic, suitable for racing, but with elements that appeal to children. Use bright guitar riffs, an upbeat tempo around 140 BPM, cheerful keyboard melodies, and add fun sound effects, such as car honks or engine noises. The music should be energetic but not aggressive, maintaining a light and playful vibe for a kids' audience", forward_params={"do_sample": True})

scipy.io.wavfile.write("musicgen_out10.wav", rate=music["sampling_rate"], data=music["audio"])
