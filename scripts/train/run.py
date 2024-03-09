from pyexpat import model
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from torch.utils.data import DataLoader

def run(args):
    # load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.add_special_tokens({'sep_token': '<SEP>'})
    tokenizer.add_special_tokens({'pad_token': '<PAD>'})
    
    # define a collate function
    def collate_fn(batch):
        text = [x["text"] for x in batch]
        text = tokenizer(text, return_tensors="pt", padding=True)
        text["labels"] = text["input_ids"].clone()
        return text

    # create a dataloader
    from mudslide.data import SentenceDataset
    dataset = SentenceDataset()
    dataloader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn)

    # load the model
    model = AutoModelForCausalLM.from_pretrained(args.model)

    # define the training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=8,
        max_steps=10000,
        save_total_limit=2,
        # fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataloader,
        tokenizer=tokenizer,
    )

    trainer.train()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2")
    args = parser.parse_args()
    run(args)