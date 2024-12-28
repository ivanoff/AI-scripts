from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import pipeline

# Загрузка дообученной модели
model = GPT2LMHeadModel.from_pretrained("./coffee_agent")
tokenizer = GPT2Tokenizer.from_pretrained("./coffee_agent")

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

#query = "As coffee-agent, tell me about new coffee"
query = "As coffee-agent, tell me about a flat white with oat milk"
if query.lower().startswith("as coffee-agent"):
    prompt = query.replace("As coffee-agent,", "").strip()
    result = generator(prompt, max_length=150, num_return_sequences=1)
    print(result[0]['generated_text'])

