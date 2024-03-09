import random
from torch.utils.data import IterableDataset
from datasets import load_dataset
from .sentence import sample

def get_dataset():
    dataset = load_dataset('osunlp/SMolInstruct', tasks=["forward_synthesis"])
    dataset = dataset["train"]
    return dataset

def process_single(record):
    fro = record["raw_input"]
    to = record["raw_output"]
    if "." in fro:
        fro = max(fro.split("."), key=len)
    return fro, to

def sample_sentence_from_record(fro, to, prob_ignore=0.0):
    if random.uniform(0, 1) < prob_ignore:
        to = None
    sentence = sample(fro, to)
    return sentence

def sample_sentence(dataset):
    record = random.choice(dataset)
    fro, to = process_single(record)
    return sample_sentence_from_record(fro, to)

def create_dataset(dataset=None, size=10000):
    if dataset is None:
        dataset = get_dataset()
    sentences = [str(sample_sentence(dataset)) for _ in range(size)]
    return sentences



