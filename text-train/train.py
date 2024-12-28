"""
This script trains a language model using the Hugging Face Transformers library and PEFT (Parameter-Efficient Fine-Tuning).

Modules:
    torch: PyTorch library for tensor computations and deep learning.
    datasets: Library for handling datasets.
    transformers: Hugging Face library for state-of-the-art NLP models.
    peft: Library for parameter-efficient fine-tuning.

Functions:
    create_prompt(agent, question, answer): Creates a formatted prompt for training data.

Configuration:
    bnb_config: Configuration for model quantization using BitsAndBytes.
    model_name: Name of the pre-trained model to be used.
    model: Loaded pre-trained model with quantization.
    tokenizer: Tokenizer for the pre-trained model.

Training Data:
    training_data: List of dictionaries containing training prompts and responses.
"""

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model
)

# Конфигурация для квантизации модели
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

# Загрузка модели и токенизатора
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Подготовка данных для обучения
def create_prompt(agent, question, answer):
    return f"<s>[INST] Agent: {agent}\nQuestion: {question} [/INST] {answer}</s>"

training_data = [
   {
        "text": create_prompt(
            "helper-agent",
            "Creating Trees",
            "Yo buddy! Just plant them. Peace!"
        )
    },
    # Add more training data here
]

dataset = Dataset.from_list(training_data)

def tokenize_function(examples):
    results = tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding=False,
        return_tensors=None
    )
    return results

tokenized_dataset = dataset.map(
    tokenize_function,
    remove_columns=dataset.column_names,
    desc="Tokenizing dataset"
)

# Создаем data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Конфигурация LoRA
lora_config = LoraConfig(
#    r=8,
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
#    lora_dropout=0.05,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# Подготовка модели для обучения
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# Параметры обучения
training_args = TrainingArguments(
    output_dir="./helper_agent_mistral",
#    num_train_epochs=3,
###
    num_train_epochs=5,
    per_device_train_batch_size=1,  # Уменьшен размер batch
    gradient_accumulation_steps=8,   # Увеличено количество шагов аккумуляции
    save_steps=100,
    logging_steps=10,
#    learning_rate=2e-4,
###
    learning_rate=1e-4,
    weight_decay=0.01,
    fp16=True,
    warmup_steps=50,
    save_total_limit=3,
    remove_unused_columns=False,     # Важно для правильной обработки данных
###
    group_by_length=True,
    lr_scheduler_type="cosine",
)

# Инициализация тренера
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# Запуск обучения
trainer.train()

# Сохранение обученной модели
model.save_pretrained("./helper_agent_mistral_final")
tokenizer.save_pretrained("./helper_agent_mistral_final")
