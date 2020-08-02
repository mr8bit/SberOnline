from django.shortcuts import render
from django.template.response import TemplateResponse
# Create your views here.
import re
import pymorphy2
morph = pymorphy2.MorphAnalyzer()
from . import reader as dataset
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from .my_layers import Attention, Average, WeightedSum, WeightedAspectEmb, MaxMargin
import keras.backend as K
from keras.preprocessing import sequence
import re
from keras.models import load_model
from . import utils as U
num_regex = re.compile('^[+-]?[0-9]+\.?[0-9]*$')
import numpy as np

def is_number(token):
    return bool(num_regex.match(token))


def one_word_preprocess(line, maxlen=256):
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

vocab, train_x, _, overall_maxlen = dataset.get_data('appstore_', maxlen=256 ,vocab_size=99507)
test_x = train_x
test_x = sequence.pad_sequences(test_x, maxlen=overall_maxlen)
out_dir = 'output' + '/' + 'appstore'

print("Load model...")
model = load_model(out_dir + '/model_param.h5',
                   custom_objects={"Attention": Attention, "Average": Average, "WeightedSum": WeightedSum,
                                   "MaxMargin": MaxMargin, "WeightedAspectEmb": WeightedAspectEmb,
                                   "max_margin_loss": U.max_margin_loss})
print("Model has been loaded...")

vocab_inv = {}
for w, ind in vocab.items():
    vocab_inv[ind] = w

splits = []
def main(request):
    context = {}
    if request.POST:
        review = request.POST['review']
        # Тут код для обработки review
        remove_shit = re.sub("[^0-9a-zA-Zа-яА-Я]", "", review)
        if len(remove_shit.split(' ')) <= 5:
            context['teams'] = ['Слишком мало слов, пиши больше']
            return TemplateResponse(request, 'review.html', context)
        else:
            tokens = parseSentence(remove_shit)
            if len(tokens) > 1:
                tokens_sentecess = ' '.join(tokens)
                sentence = one_word_preprocess(tokens_sentecess)
                test_fn = K.function([model.get_layer('sentence_input').input, K.learning_phase()],
                                     [model.get_layer('att_weights').output, model.get_layer('p_t').output])
                att_weights, aspect_probs = [], []
                cur_att_weights, cur_aspect_probs = test_fn([sentence, 0])
                att_weights.append(cur_att_weights)
                aspect_probs.append(cur_aspect_probs)
                label_ids = np.argsort(aspect_probs, axis=1)[:, -1]
                print(label_ids)
            else:
                context['teams'] = ['Слишком мало слов, пиши больше']
                return TemplateResponse(request, 'review.html', context)




        context['teams'] = ['Команды', 'iOS Platform', 'Android Team Development']
        return TemplateResponse(request, 'review.html', context)
    return TemplateResponse(request, 'index.html', context)