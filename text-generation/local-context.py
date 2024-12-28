"""
This script uses the Llama model to process text files in a specified directory and generate a response to a given question.

Modules:
    llama_cpp (Llama): A module for interacting with the Llama model.
    os: A module providing a way of using operating system dependent functionality.

Constants:
    model_path (str): The path to the Llama model file.
    docs_dir (str): The directory containing the text files to be processed.

Variables:
    llm (Llama): An instance of the Llama model.
    content (str): A string to accumulate the content of all text files.
    docs (list): A list to store Document objects (currently commented out).

Functions:
    None

Usage:
    The script reads all text files in the specified directory, concatenates their content, and uses the Llama model to generate a response to a predefined question.
"""

from llama_cpp import Llama
import os

model_path = "./hf_bartowski_gemma-2-27b-it-Q4_K_L.gguf"
#model_path = "./hf_mradermacher_Llama-3.2-3B-Instruct.Q8_0.gguf"
#model_path = "./hf_mradermacher_Mistral-Nemo-Instruct-2407.Q8_0.gguf"
#model_path = "./hf_mradermacher_Mistral-Nemo-Instruct-2407.Q6_K.gguf"

llm = Llama(
    model_path=model_path,
    n_ctx=12048,
#    n_threads=8,
)

content = ""
docs_dir = "./docs"
docs = []

for filename in os.listdir(docs_dir):
    filepath = os.path.join(docs_dir, filename)
    if os.path.isfile(filepath):
        with open(filepath, 'r', encoding='utf-8') as file:
#            content = file.read()
#            docs.append(Document(content=content))

            content += file.read() + "\n"

#docs = [Document(content=content)]

response = llm("Context: " + content + "Question: how to create asset?")
print(response["choices"][0]["text"])
print(response)
