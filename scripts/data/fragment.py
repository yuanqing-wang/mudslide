import pandas as pd
from mudslide.data import get_dataset, process_single
from mudslide.sentence import sample
from multiprocessing import Pool
import tqdm

def process_record(record):
    try:
        fro, to = process_single(record)
        sentence = sample(fro)
        return str(sentence)
    except:
        return None

def run(args):
    dataset = get_dataset()
    sentences = []

    pool = Pool(4)
    # sentences = tqdm.tqdm(pool.imap(process_record, dataset), total=len(dataset))

    for record in tqdm.tqdm(dataset):
        sentence = process_record(record)
        print(sentence)
        sentences.append(sentence)

    setences = [s for s in sentences if s is not None]
    df = pd.DataFrame(sentences, columns=["text"])
    df.to_csv(args.output, index=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="fragment.csv")
    args = parser.parse_args()
    run(args)
