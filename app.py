from flask import Flask, jsonify, request, render_template, make_response, send_file
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import numpy as np
import string
from bert import tokenization
import time
import pickle
LEVELS = 3
MAX_SEQUENCE_LENGTH = 30

def text_process(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join([c for c in nopunc if not c.isdigit()])
    nopunc = nopunc.replace(u'\xa0', u'')
    nopunc = nopunc.strip()
    return nopunc


bert_layer = hub.KerasLayer("bert_multi_cased_L-12_H-768_A-12_3",
                            trainable=True)
vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocabulary_file, to_lower_case)

def tokenize_text(v):
    return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(v))

app = Flask(__name__, template_folder='templates')

@app.route('/')
def index():
    return render_template('predict.html')

@app.route('/predict-one', methods = ['POST'])
def model_predict():
    txt = request.form["txt"]
    results = {}
    text = text_process(txt)
    text = tokenize_text(text)
    text = pad_sequences([text], maxlen=MAX_SEQUENCE_LENGTH)
    model = tf.keras.models.load_model('pruned_models/model_lvl_1.h5')
    result = model.predict(text)
    pkl_file = open('labelEncoders/le_lvl_1.pkl', 'rb')
    le = pickle.load(pkl_file)
    pkl_file.close()
    result = le.inverse_transform(np.argmax(result, axis=-1))
    results['lvl1'] = result[0]
    for i in range(2, LEVELS + 1):
        lvl = 'lvl' + str(i)
        print(result[0])
        pkl_file_ = open('labelEncoders/le_lvl_' + str(i) + '_' + str(result[0]) + '.pkl', 'rb')
        le_ = pickle.load(pkl_file_)
        pkl_file.close()
        text_model = tf.keras.models.load_model('pruned_models/model_lvl_' + str(i) + '_' + str(result[0]) + '.h5')
        print(text)
        result = text_model.predict(text)
        print(result)
        result = le_.inverse_transform(np.argmax(result, axis=-1))
        if result[0] != "nan":
            results[lvl] = result[0]
        tf.keras.backend.clear_session()
    return jsonify({'results: ': results})


@app.route('/predict-all', methods=['POST'])
def model_predict_all():
    f = request.files['data_file']
    if not f:
        return "No file"
    start = time.time()
    df = pd.read_csv(f, skiprows=1, names=["raw"], encoding='utf8')
    temp = [tokenize_text(t) for t in df["raw"]]
    tmp = pad_sequences(temp, maxlen=MAX_SEQUENCE_LENGTH)
    model = tf.keras.models.load_model('pruned_models/model_lvl_1.h5')
    result = model.predict_on_batch(tmp)
    pkl_file = open('labelEncoders/le_lvl_1.pkl', 'rb')
    le = pickle.load(pkl_file)
    pkl_file.close()
    result = le.inverse_transform(np.argmax(result, axis=-1))
    df["lvl1"] = result
    tf.keras.backend.clear_session()
    for i in range(2, LEVELS+1):
        result_ = []
        lvl = 'lvl' + str(i)
        for j in range(len(result)):
            pkl_file_ = open('labelEncoders/le_lvl_' + str(i) + '_' + str(result[j]) + '.pkl', 'rb')
            le_ = pickle.load(pkl_file_)
            pkl_file.close()
            text_model = tf.keras.models.load_model('pruned_models/model_lvl_' + str(i) + '_' + str(result[j]) + '.h5')
            tmp = pad_sequences([temp[j]], maxlen=MAX_SEQUENCE_LENGTH)
            output = text_model.predict(tmp)
            output = le_.inverse_transform(np.argmax(output, axis=-1))
            print(output)
            result_.append(output[0])
        df[lvl] = result_
        result = result_
        tf.keras.backend.clear_session()

    df.to_csv("export.csv", index=False, sep=",", encoding='utf-8-sig')
    total_time = time.time() - start
    print("Time: ", total_time)
    return send_file("export.csv", as_attachment=True,
              attachment_filename='export.csv',
              mimetype='text/csv')




if __name__ == '__main__':
    app.run(debug= True)