from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import torch

def main():
    # Load the Solana dataset from local file
    print("Loading dataset...")
    dataset = load_dataset('json', data_files={'train': 'solana_1000.json'})
    print("\nDataset structure:")
    print(dataset)
    print("\nFirst training example:")
    print(dataset["train"][0])

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Prepare and tokenize the dataset
    print("\nPreparing and tokenizing dataset...")
    def preprocess_function(examples):
        # Combine instruction, input, and output into a single text
        texts = []
        for i in range(len(examples['instruction'])):
            instruction = examples['instruction'][i]
            inp = examples['input'][i]
            output = examples['output'][i]
            
            # Format: "Instruction: {instruction}\nInput: {input}\nOutput: {output}"
            text = f"Instruction: {instruction}\n"
            if inp:  # Only add input if it's not empty
                text += f"Input: {inp}\n"
            text += f"Output: {output}"
            texts.append(text)
        
        # Tokenize the combined texts
        model_inputs = tokenizer(texts, truncation=True, padding="max_length", max_length=512)
        
        # Create the labels (same as input_ids for causal language modeling)
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        
        return model_inputs

    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    # Load model
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model.resize_token_embeddings(len(tokenizer))

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./solana-model",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=500,
        warmup_steps=500,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),  # Enable mixed precision training if GPU available
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
    )

    # Train model
    print("\nStarting training...")
    trainer.train()

    # Save model
    print("\nSaving model...")
    model.save_pretrained("./solana-model")
    tokenizer.save_pretrained("./solana-model")
    print("\nTraining complete! Model saved to ./solana-model")

if __name__ == "__main__":
    main()
