"""
This script provides functionality to load a pre-trained language model with a LoRA adapter, create prompts, and generate responses to user questions.
Functions:
    load_helper_agent(base_model_name="mistralai/Mistral-7B-Instruct-v0.2", adapter_path="./helper_agent_mistral_final"):
        Loads the base model and LoRA adapter, and returns the model and tokenizer.
    create_prompt(question):
        Creates a prompt in the Mistral format for the given question.
    get_helper_recommendation(model, tokenizer, question, max_length=512):
        Generates a recommendation from the helper-agent based on the given question.
    main():
        Main function to load the model, prompt the user for questions, and print the helper-agent's responses.
"""
s
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

def load_helper_agent(base_model_name="mistralai/Mistral-7B-Instruct-v0.2", 
                     adapter_path="./helper_agent_mistral_final"):
    """
    Загрузка модели с адаптером LoRA
    """
    # Загрузка конфигурации и базовой модели
    config = PeftConfig.from_pretrained(adapter_path)
    
    # Загрузка токенизатора
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    # Загрузка базовой модели
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Загрузка обученных весов LoRA
    model = PeftModel.from_pretrained(model, adapter_path)
    
    return model, tokenizer

def create_prompt(question):
    """
    Создание промпта в формате Mistral
    """
    return f"<s>[INST] Agent: helper-agent\nQuestion: {question} [/INST]"

def get_helper_recommendation(model, tokenizer, question, max_length=512):
    """
    Получение рекомендации от helper-agent
    """
    # Подготовка промпта
    prompt = create_prompt(question)
    
    # Токенизация
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Генерация ответа
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.8,  # Увеличиваем температуру для более "живых" ответов
            top_p=0.92,
            top_k=50,  # Добавляем top_k для лучшего контроля стиля
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2  # Добавляем штраф за повторения
#            **inputs,
#            max_length=max_length,
#            num_return_sequences=1,
#            temperature=0.7,
#            top_p=0.95,
#            do_sample=True,
#            pad_token_id=tokenizer.eos_token_id
        )
    
    # Декодирование ответа
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Извлечение только ответа (после [/INST])
    answer = response.split("[/INST]")[-1].strip()
    
    return answer

def main():
    # Загрузка модели и токенизатора
    print("Загрузка helper-agent...")
    model, tokenizer = load_helper_agent()
    
    print("helper Agent готов! Задавайте вопросы о кофе (для выхода введите 'exit')")
    
    while True:
        question = input("\nВаш вопрос: ")
        
        if question.lower() == 'exit':
            print("До свидания!")
            break
            
        try:
            answer = get_helper_recommendation(model, tokenizer, question)
            print(f"\nhelper Agent: {answer}")
        except Exception as e:
            print(f"Произошла ошибка: {str(e)}")

if __name__ == "__main__":
    main()
