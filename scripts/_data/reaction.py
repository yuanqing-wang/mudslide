import pandas as pd
from mudslide.data import get_dataset, process_single
from mudslide.sentence import sample
from multiprocessing import Pool

def process_record(record):
    fro, to = process_single(record)
    sentence = sample(fro, to)
    return str(sentence)

def run(args):
    dataset = get_dataset()
    sentences = []

    with Pool() as pool:
        sentences = pool.map(process_record, dataset)

    df = pd.DataFrame(sentences, columns=["text"])
    df.to_csv(args.output, index=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="reaction.csv")
    args = parser.parse_args()
    run(args)
