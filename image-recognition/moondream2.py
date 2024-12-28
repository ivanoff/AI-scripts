"""
This script uses a pre-trained language model to describe the content of an image.

Modules:
    transformers: Provides the AutoModelForCausalLM and AutoTokenizer classes for loading the pre-trained model and tokenizer.
    PIL: Provides the Image class for opening and manipulating images.

Constants:
    model_id (str): The identifier for the pre-trained model.
    revision (str): The specific revision of the model to use.
    image_path (str): The file path to the image to be analyzed.

Functions:
    encode_image(image): Encodes the image using the pre-trained model.
    answer_question(encoded_image, question, tokenizer): Uses the pre-trained model to answer a question about the encoded image.

Usage:
    The script loads a pre-trained model and tokenizer, opens an image, encodes the image, and then uses the model to answer a question about the image. The response is printed to the console.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

model_id = "vikhyatk/moondream2"
revision = "2024-08-26"

model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, revision=revision
)
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

image_path = "./ba.png"
image = Image.open(image_path)

enc_image = model.encode_image(image)
response = model.answer_question(enc_image, "Describe this image.", tokenizer)
#response = model.answer_question(enc_image, "Does this image contains sexual context?", tokenizer)
print(response)
