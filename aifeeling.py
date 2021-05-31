import numpy as np
from keras.models import load_model
from konlpy.tag import Okt
import json

okt = Okt()


def tokenize(doc):
    #형태소와 품사를 join
    return ['/'.join(t) for t in okt.pos(doc, norm=True, stem=True)]


with open("selected.json", "r") as f:
    selected_words = json.load(f)

selected_words = selected_words["text"]


#단어리스트 문서에서 상위 10000개들중 포함되는 단어들이 개수
def term_frequency(doc):
    return [doc.count(word) for word in selected_words]


model = load_model('movie_review_model.h5')

review = "아주 재미 있어요"
token = tokenize(review)

tf = term_frequency(token)
data = np.expand_dims(np.asarray(tf).astype('float32'), axis=0)
float(model.predict(data))


def predict_review(review):
    token = tokenize(review)
    tfq = term_frequency(token)
    data = np.expand_dims(np.asarray(tfq).astype('float32'), axis=0)
    score = float(model.predict(data))
    if (score > 0.5):
        print(f"{review} ==> 긍정 ({round(score*100)}%)")
    else:
        print(f"{review} ==> 부정 ({round((1-score)*100)}%)")


predict_review("ㅋ.ㅋ")