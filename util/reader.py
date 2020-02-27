import pandas as pd


def read_test_tsv(path):
    id = []
    tweet = []
    with open(path, 'rt') as file_in:
        next(file_in)
        for line in file_in:
            line = line.replace("\n", "")
            components = line.split(sep="\t")
            id.append(components[0])
            tweet.append(components[1])

    return pd.DataFrame(
        {'id': id,
         'tweet': tweet
         })


def read_train_tsv(path):
    id = []
    tweet = []
    label = []
    with open(path, 'rt') as file_in:
        next(file_in)
        for line in file_in:
            line = line.replace("\n", "")
            components = line.split(sep="\t")
            id.append(components[0])
            tweet.append(components[1])
            label.append(components[2])

    return pd.DataFrame(
        {'id': id,
         'tweet': tweet,
         'subtask_a': label
         })
