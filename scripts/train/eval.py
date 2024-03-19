import torch
import pandas as pd
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from datasets import Dataset
from sklearn.metrics import accuracy_score
# from peft import PeftConfig, PeftModel


def run():
    model_name = "results/checkpoint-10000"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Evaluate the model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))

    # Sample string
    sample_string = "Generate analogs of C.<SEP>"

    # Tokenize the sample string
    tokenized_input = tokenizer(
        [sample_string], return_tensors="pt", padding=True, 
    )

    # Generate output using the model
    with torch.no_grad():
        outputs = model.generate(
            tokenized_input["input_ids"],
            max_new_tokens=500,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode the generated output
    generated_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print("Generated Output:", generated_output)


if __name__ == "__main__":
    run()

