"""
This script demonstrates the usage of the Llama model for various natural language processing tasks such as translation, paraphrasing, and context-based JSON value filling.

Imports:
    - Llama from llama_cpp

Variables:
    - model_path (str): Path to the model file.

Model Initialization:
    - Initializes the Llama model with the specified model path and context size (n_ctx).

Functions:
    - Translation: Translates the given English text to Russian.
    - Paraphrasing: Paraphrases the given English text.
    - Context-based JSON Value Filling: Fills empty values in a JSON object with Russian translations based on the provided context.

Example Usage:
    - Translates "Hey, buddy! What's up! Did you received my last message?" to Russian.
    - Paraphrases "Hey, buddy! What's up! Did you received my last message?".
    - Fills empty values in a JSON object with Russian translations based on the context of ships and vessels.
"""

from llama_cpp import Llama

model_path = "./hf_bartowski_gemma-2-27b-it-Q4_K_L.gguf"
#model_path = "./hf_mradermacher_Llama-3.2-3B-Instruct.Q8_0.gguf"
#model_path = "./hf_mradermacher_Mistral-Nemo-Instruct-2407.Q8_0.gguf"
#model_path = "./hf_mradermacher_Mistral-Nemo-Instruct-2407.Q6_K.gguf"

llm = Llama(
    model_path=model_path,
    n_ctx=2048,
#    n_threads=8,
)

response = llm("translate to ru: Hey, buddy! What's up! Did you received my last message?")
print(response["choices"][0]["text"])
print(response)

response = llm("paraphrase: Hey, buddy! What's up! Did you received my last message?")
print(response["choices"][0]["text"])

response = llm("context: ships and vessels. I have json with keys in english and empty values. Fill empty values with translated to ru. Return result as json: { 'Select request type*': ', 'I want to buy': ', 'I want to sell': ', 'I have an open cargo': ', 'I have an open ship': ', 'Select ship type*': ', 'Anchor Handling Tug Supply (AHTS)': ', 'Fast Supply Vessel (FSV)': ', 'Survey': ', 'Work boats': ', 'Tuna Longliners': ', 'Beam Trawler': ', 'Newbuild Vessels': ', 'Title*': ', 'Free-form message': ', 'Type': ', 'Capesize': ', 'sea': ', 'sea-river': ', 'Hull': ', 'DWT': ', 'GRT': ', 'NRT': ', 'LOA': ', 'LBP': ', 'Depth': ', 'Ice class': ', 'Crane cap.': ', 'Crane rev.': ', 'Passengers': ', 'Decks': ', 'Teu': ', 'Cars': ', 'Main Engine': ', 'Type of fuel': ', 'DD last': ', 'DD next': ', 'SS last': ', 'SS next': ', 'Asphalt carrier': ', 'Attach a file': ', 'August': ', 'Australasia': ', 'Auxiliary engine': ', 'Average reefer': ', 'Baltiyskiy': ', 'Barge': ', 'Beam': ', 'Black Sea': ', 'box shaped': ', 'Build year': ', 'Built in': ', 'built year': ', 'Bulk carrier': ', 'Bunkering vessel': ', 'Cable layer': ', 'Capacity': ', 'Car float': '}")
print(response)
