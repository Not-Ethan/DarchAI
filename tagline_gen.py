import datasets
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split


# Preprocessing function
def preprocess_function(examples, tokenizer):
    input_texts = examples["Extract"]
    target_texts = examples["Abstract"]
    
    model_inputs = tokenizer(input_texts, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(target_texts, max_length=512, truncation=True, padding="max_length")
    
    model_inputs["labels"] = labels["input_ids"]
    
    return model_inputs

def load_preprocess_data(tokenizer, split="train"):
    dataset = datasets.load_dataset("csv", data_files="/kaggle/input/debateall/debateall.csv")
    data = dataset["train"]
    
    data = data.shuffle(seed=42).select(range(int(len(data) * 0.4)))
    
    # Split the data into train and validation sets
    train_data, val_data = train_test_split(data, test_size=0.1, random_state=42)

    train_dataset = datasets.Dataset.from_dict(train_data)
    val_dataset = datasets.Dataset.from_dict(val_data)

    if split == "train":
        preprocessed_data = train_dataset.map(lambda examples: preprocess_function(examples, tokenizer), batched=True)
    elif split == "validation":
        preprocessed_data = val_dataset.map(lambda examples: preprocess_function(examples, tokenizer), batched=True)
    
    preprocessed_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return preprocessed_data

# Train summarization model
def train_summarization_model():

    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    train_data = load_preprocess_data(tokenizer, split="train")
    val_data = load_preprocess_data(tokenizer, split="validation")
    
    training_args = TrainingArguments(
        output_dir="/kaggle/working/models",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy="steps",
        gradient_accumulation_steps=2,
        eval_steps=100,
        save_steps=100,
        logging_steps=100,
        logging_dir="/kaggle/working/logs",
        load_best_model_at_end=True,
        save_total_limit = 2,
        fp16=True

    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data
    )
    
    trainer.train()

# Run the training
train_summarization_model()
