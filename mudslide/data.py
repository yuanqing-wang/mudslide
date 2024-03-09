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

def data_generator():
    dataset = get_dataset()
    while True:
        yield {"text": str(sample_sentence(dataset))}

class SentenceDataset(IterableDataset):
    def __init__(self, dataset=None):
        if dataset is None:
            dataset = data_generator()
        self.dataset = dataset

    def __iter__(self):
        return iter(self.dataset)
    
    def __len__(self):
        return 99999
    
    





