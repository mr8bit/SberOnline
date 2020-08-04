import sys
import pandas as pd


def distance(s1, s2):
    d = {}
    lenstr1 = len(s1)
    lenstr2 = len(s2)
    for i in range(-1, lenstr1 + 1):
        d[(i, -1)] = i + 1
    for j in range(-1, lenstr2 + 1):
        d[(-1, j)] = j + 1

    for i in range(lenstr1):
        for j in range(lenstr2):
            if s1[i] == s2[j]:
                cost = 0
            else:
                cost = 1
            d[(i, j)] = min(
                d[(i - 1, j)] + 1,  # deletion
                d[(i, j - 1)] + 1,  # insertion
                d[(i - 1, j - 1)] + cost,  # substitution
            )
            if i and j and s1[i] == s2[j - 1] and s1[i - 1] == s2[j]:
                d[(i, j)] = min(d[(i, j)], d[i - 2, j - 2] + cost)  # transposition

    return d[lenstr1 - 1, lenstr2 - 1]

def get_answer(text, df=None):
    result = ""
    if df is None:
        df = pd.read_csv('./main/some_ml/answer_dataset.csv', index_col=0)
    df = df[:15000]
    dist = sys.maxsize
    for entity in df.values:
        new_dist = distance(text[:30], entity[1][:30])
        if new_dist <= dist:
            dist = new_dist
            result = entity[0]
    return result

