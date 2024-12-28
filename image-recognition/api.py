"""
API for processing images and answering questions using a pre-trained language model.

This module sets up a Flask web server that accepts POST requests with an image file and a question string.
It uses a pre-trained language model to process the image and generate an answer to the question.

Environment Variables:
- API_TOKEN: The token required to authorize API requests.

Routes:
- POST /: Processes the image and answers the question.

Request Headers:
- Authorization: Bearer token for API authorization.

Request Form Data:
- image_file: The image file to be processed.
- question_string: The question string to be answered.

Response:
- JSON object containing the question and the generated answer, or an error message.

Raises:
- ValueError: If API_TOKEN is not set in the .env file.
"""
import os
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
API_TOKEN = os.getenv("API_TOKEN")

if not API_TOKEN:
    raise ValueError("API_TOKEN is not set in the .env file")

# Initialize model and tokenizer
model_id = "vikhyatk/moondream2"
revision = "2024-08-26"
model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, revision=revision
)
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

# Create Flask app
app = Flask(__name__)

@app.route("/", methods=["POST"])
def process_image():
    # Check API token
    token = request.headers.get("Authorization")
    if not token or token.split(" ")[-1] != API_TOKEN:
        return jsonify({"error": "Unauthorized"}), 401

    # Check form data
    if "image_file" not in request.files or "question_string" not in request.form:
        return jsonify({"error": "Both 'image_file' and 'question_string' are required"}), 400

    image_file = request.files["image_file"]
    question_string = request.form["question_string"]

    try:
        # Load and process the image
        image = Image.open(image_file)
        enc_image = model.encode_image(image)

        # Generate the answer
        answer = model.answer_question(enc_image, question_string, tokenizer)
        return jsonify({"question": question_string, "answer": answer}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

