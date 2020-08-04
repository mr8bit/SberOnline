import pymorphy2
from sklearn.feature_extraction.text import CountVectorizer
import re
from nltk.corpus import stopwords

num_regex = re.compile('^[+-]?[0-9]+\.?[0-9]*$')
morph = pymorphy2.MorphAnalyzer()


def get_with_threshold(inputs, threshold, top=None):
    result = []
    labeled_inputs = dict(zip(range(len(inputs)), inputs))
    filtered = dict(filter(lambda x: x if x[1] <= threshold else None, labeled_inputs.items()))
    sorted_filtered = {k: v for k, v in sorted(filtered.items(), key=lambda item: item[1])}
    result = list(sorted_filtered.keys())
    result.reverse()
    if top is not None:
        result = result[:top]
    return result


def is_number(token):
    return bool(num_regex.match(token))


def one_word_preprocess(line, vocab, maxlen=256):
    num_hit, unk_hit, total = 0., 0., 0.
    maxlen_x = 0
    data_x = []

    words = line.strip().split()
    if maxlen > 0 and len(words) > maxlen:
        words = words[:maxlen]

    indices = []
    for word in words:
        if is_number(word):
            indices.append(vocab['<num>'])
            num_hit += 1
        elif word in vocab:
            indices.append(vocab[word])
        else:
            indices.append(vocab['<unk>'])
            unk_hit += 1
        total += 1

    data_x.append(indices)
    if maxlen_x < len(indices):
        maxlen_x = len(indices)
    return data_x, maxlen_x


def parseSentence(line):
    stop = stopwords.words('russian')
    text_token = CountVectorizer().build_tokenizer()(line.lower())
    text_rmstop = [i for i in text_token if i not in stop]
    text_stem = [morph.parse(w)[0].normal_form for w in text_rmstop]
    return text_stem
