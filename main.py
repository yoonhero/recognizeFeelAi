from re import L
from flask import request
from flask import Flask, render_template, render_template_string, Response, jsonify
import numpy as np
from keras.models import load_model
from konlpy.tag import Okt
import json
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

okt = Okt()
model = load_model('movie_review_model.h5')


def tokenize(doc):
    #형태소와 품사를 join
    return ['/'.join(t) for t in okt.pos(doc, norm=True, stem=True)]


def term_frequency(doc):
    return [doc.count(word) for word in selected_words]


def predict_review(review):
    token = tokenize(review)
    tfq = term_frequency(token)
    data = np.expand_dims(np.asarray(tfq).astype('float32'), axis=0)
    score = float(model.predict(data))
    return score


with open("selected.json", "r") as f:
    selected_words = json.load(f)

selected_words = selected_words["text"]

app = Flask(__name__)


@app.route('/')
def feeling():
    ok = True
    error = ""
    temp = request.args.get("text")
    score = 0
    print(temp)
    try:
        score = predict_review(temp)
    except:
        ok = False
        error = "Internal Server error"
    return jsonify({'ok': ok, 'error': error, "feeling": round(score * 100)})


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000)
