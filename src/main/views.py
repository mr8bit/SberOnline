from django.shortcuts import render
from django.template.response import TemplateResponse
# Create your views here.
import re

from . import reader as dataset
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from .my_layers import Attention, Average, WeightedSum, WeightedAspectEmb, MaxMargin
import keras.backend as K
from keras.preprocessing import sequence
from keras.models import load_model
from . import utils as U
from main.some_ml.utils import parseSentence, one_word_preprocess, get_with_threshold
import numpy as np
from main.some_ml.auto_answer import get_answer

out_dir = './main/output' + '/' + 'appstore'

print("Load vocab...")
vocab = dataset.load_only_vocab('appstore', maxlen=256, vocab_size=99507)
print("Complete load vocab...")

print("Load model...")
model = load_model(out_dir + '/model_param.h5',
                   custom_objects={"Attention": Attention, "Average": Average,
                                   "WeightedSum": WeightedSum,
                                   "MaxMargin": MaxMargin, "WeightedAspectEmb": WeightedAspectEmb,
                                   "max_margin_loss": U.max_margin_loss})
print("Model has been loaded...")
test_fn = K.function([model.get_layer('sentence_input').input, K.learning_phase()],
                     [model.get_layer('att_weights').output, model.get_layer('p_t').output])


splits = []
def main(request):
    context = {}
    if request.POST:
        review = request.POST['review']
        # Тут код для обработки review
        remove_shit = re.sub("[^0-9a-zA-Zа-яА-Я]", " ", review)
        if len(remove_shit.split(' ')) <= 5:
            context['teams'] = ['Слишком мало слов, пиши больше']
            return TemplateResponse(request, 'review.html', context)
        else:
            tokens = parseSentence(remove_shit)
            print(tokens)
            if len(tokens) > 1:
                tokens_sentecess = ' '.join(tokens)
                vec, maxlen_x = one_word_preprocess(tokens_sentecess, vocab)
                sentence = sequence.pad_sequences(vec, maxlen=256)
                att_weights, aspect_probs = [], []
                print(sentence)
                cur_att_weights, cur_aspect_probs = test_fn([sentence, 0])
                aspect_probs.append(cur_aspect_probs)
                print(aspect_probs)
                labels_ids = get_with_threshold(aspect_probs[0][0], 1, 2)
                answer = get_answer(remove_shit)
                context['teams'] = []
                context['answer'] = answer
                for team in labels_ids:
                    if team == 125:
                        context['teams'].append('iOS Platform')
                    if team == 60:
                        context['teams'].append('iOS Release Engineer')
                    if team == 92:
                        context['teams'].append('История операций')
                    if team == 62:
                        context['teams'].append('СБОЛ. Классические переводы')
                    if team == 107:
                        context['teams'].append('PUSH iOS')
                    if team == 111:
                        context['teams'].append('ВС.МП вклады')
                    if team == 112:
                        context['teams'].append('Digital Сбербанк Премьер')
                    if team == 8:
                        context['teams'].append('Дезигн')
                return TemplateResponse(request, 'review.html', context)
            else:
                context['teams'] = ['Слишком мало слов, пиши больше']
                return TemplateResponse(request, 'review.html', context)

    return TemplateResponse(request, 'index.html', context)
