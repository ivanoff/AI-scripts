from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# Загрузка модели и токенизатора
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained(model_name)

# Подготовка данных
dataset = load_dataset('text', data_files={'train': 'your_train_data.txt'})

#def tokenize_function(examples):
#    return tokenizer(examples['text'], truncation=True, padding=True, max_length=512)

#gpt2 only
def tokenize_function(examples):
    # Токенизация с добавлением меток
    outputs = tokenizer(
        examples['text'], 
        truncation=True, 
        padding=True, 
        max_length=512
    )
    # Копируем input_ids в labels и заменяем pad_token_id на -100
    outputs["labels"] = [
        [(token if token != tokenizer.pad_token_id else -100) for token in ids]
        for ids in outputs["input_ids"]
    ]
    return outputs

tokenized_dataset = dataset.map(tokenize_function, batched=True)

#tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Настройка аргументов обучения
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
)

# Тренировка
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
)

trainer.train()
model.save_pretrained("./coffee_agent")
tokenizer.save_pretrained("./coffee_agent")

