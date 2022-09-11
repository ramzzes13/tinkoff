import argparse, re
from collections import defaultdict, Counter
import numpy as np
import numpy.random
import pandas as pd
import random


def GenerateNGramms(text, n=2):
    return np.array([text[i: i + n] for i in range(0, text.size - n)])


def get_index(word, vocab):
    return np.where(vocab == word)[0][0]


def predict(word, freq_table, text, unique_sorted):
    w_index = get_index(word, unique_sorted)
    ans = []
    predictions = []
    w_freq = numpy.sum(freq_table[w_index]) - freq_table.shape[0]
    for i in range(freq_table.shape[0]):
        indices = tuple([w_index, get_index(unique_sorted[i], unique_sorted)])
        prob = freq_table[indices] / (w_freq + 1)
        if freq_table[indices] == 1:
            prob = 0
        predictions.append(prob)

        # print(word, unique_sorted[i], prob)
    mx = max(predictions)
    # print(mx)
    for i in range(freq_table.shape[0]):
        if mx - 0.02 < predictions[i] < mx + 0.02:
            ans.append(unique_sorted[i])
    return ans


def train(link="./in.txt", n=2):
    text = ""
    with open(link, "r", encoding="UTF-8") as file:
        for i in file:
            text += i
    text = text.lower()
    text = re.split('[^a-zа-яё]+', text, flags=re.IGNORECASE)
    text = np.array(text, dtype=str)
    text = text[text != '']

    ngramms = GenerateNGramms(text, n)

    unique = np.unique(text)
    unique.sort()
    unique_size = unique.size

    # print(unique)

    # print(unique_size, text.size, text.size / unique_size)
    freq_table = np.array([1] * unique_size ** n, dtype=np.int64).reshape(*[unique_size] * n)

    for ngr in ngramms:
        indices = [get_index(word, unique) for word in ngr]
        freq_table[tuple(indices)] += 1
    # predict('андрей', freq_table, text, unique)
    # print(freq_table[15])

    # while True:
    #    a = input().lower()
    #    prdct = predict(a, freq_table, text, unique)
    #    print(prdct)
    a = input("Type in your word").lower()
    prdct = predict(a, freq_table, text, unique)
    print(prdct)
    


def main():
    parser = argparse.ArgumentParser(description='type dir to in.txt and n parameter ')
    parser.add_argument('input_dir', type=str, help='Input dir for in.txt')
    parser.add_argument('n', type=int, help='N for n-gram model')
    args = parser.parse_args()
    if args.input_dir == "":
        raise Exception
    train(link=args.input_dir, n=args.n)


if __name__ == "__main__":
    main()
