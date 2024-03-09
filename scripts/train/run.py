from mudslide import data
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from torch.utils.data import DataLoader
from datasets import Dataset

def run(args):
    # load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.add_special_tokens({'sep_token': '<SEP>'})
    tokenizer.add_special_tokens({'pad_token': '<PAD>'})

    def tokenize_function(examples):
        output = tokenizer(examples["text"], padding="max_length", truncation=True)
        output["labels"] = output["input_ids"]
        return output

    # load the dataset
    from mudslide.data import create_dataset
    dataset = create_dataset(size=1000)

    import pandas as pd
    dataset = pd.DataFrame(dataset, columns=["text"])
    dataset = Dataset.from_pandas(dataset)
    dataset = dataset.map(tokenize_function, batched=True)

    # load the model
    model = AutoModelForCausalLM.from_pretrained(args.model)

    # define the training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=4,
        max_steps=10000,
        save_total_limit=2,
        # fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2")
    args = parser.parse_args()
    run(args)