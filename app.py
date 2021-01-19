from flask import Flask, jsonify, request, render_template, make_response, send_file
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pandas as pd
import numpy as np
import string
import pickle
from transformers import *
from transformers import BertTokenizer, TFBertModel, BertConfig
import os
LEVELS = 3
MAX_SEQUENCE_LENGTH = 30

def text_process(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join([c for c in nopunc if not c.isdigit()])
    nopunc = nopunc.replace(u'\xa0', u'')
    nopunc = nopunc.strip()
    return nopunc

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
def tokenize_text(v):
    return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(v))

def predict_all(df):
    temp = [tokenize_text(text_process(t)) for t in df["raw"]]
    tmp = pad_sequences(temp, maxlen=MAX_SEQUENCE_LENGTH)
    model = tf.keras.models.load_model('models/model_lvl_1.h5')
    result = model.predict_on_batch(tmp)
    pkl_file = open('labelEncoders/le_lvl_1.pkl', 'rb')
    le = pickle.load(pkl_file)
    pkl_file.close()
    result = le.inverse_transform(np.argmax(result, axis=-1))
    df["lvl1"] = result
    tf.keras.backend.clear_session()
    for i in range(2, LEVELS + 1):
        result_ = []
        lvl = 'lvl' + str(i)
        for j in range(len(result)):
            pkl_file_ = open('labelEncoders/le_lvl_' + str(i) + '_' + str(result[j]) + '.pkl', 'rb')
            le_ = pickle.load(pkl_file_)
            pkl_file.close()
            text_model = tf.keras.models.load_model('models/model_lvl_' + str(i) + '_' + str(result[j]) + '.h5')
            tmp = pad_sequences([temp[j]], maxlen=MAX_SEQUENCE_LENGTH)
            output = text_model.predict(tmp)
            output = le_.inverse_transform(np.argmax(output, axis=-1))
            print(output)
            result_.append(output[0])
        df[lvl] = result_
        result = result_
        tf.keras.backend.clear_session()
    return df

app = Flask(__name__, template_folder='templates')
content = ""
filepath = ""

@app.route('/')
def index():
    return render_template('predict.html', content=content)

@app.route('/predict-one', methods = ['POST'])
def model_predict():
    txt = request.form["txt"]
    results = {}
    text = text_process(txt)
    text = tokenize_text(text)
    text = pad_sequences([text], maxlen=MAX_SEQUENCE_LENGTH)
    model = tf.keras.models.load_model('models/model_lvl_1.h5')
    result = model.predict(text)
    pkl_file = open('labelEncoders/le_lvl_1.pkl', 'rb')
    le = pickle.load(pkl_file)
    pkl_file.close()
    result = le.inverse_transform(np.argmax(result, axis=-1))
    results['lvl1'] = result[0]
    for i in range(2, LEVELS + 1):
        lvl = 'lvl' + str(i)
        pkl_file_ = open('labelEncoders/le_lvl_' + str(i) + '_' + str(result[0]) + '.pkl', 'rb')
        le_ = pickle.load(pkl_file_)
        pkl_file.close()
        text_model = tf.keras.models.load_model('models/model_lvl_' + str(i) + '_' + str(result[0]) + '.h5')
        result = text_model.predict(text)
        result = le_.inverse_transform(np.argmax(result, axis=-1))
        if result[0] == "nan":
            result[0] = " "
            ul_sub = "list-unstyled"
        results[lvl] = result[0]
        tf.keras.backend.clear_session()
    return render_template("download.html", text=txt,
                           lvl1=results['lvl1'], lvl2 = results['lvl2'], lvl3= results['lvl3'])


@app.route('/predict-all', methods=['POST'])
def model_predict_all():
    f = request.files['data_file']
    if not f:
        return "No file"

    fileName, fileExtension = os.path.splitext(f.filename)
    print(fileExtension)
    if fileExtension == '.csv':
        df = pd.read_csv(f, skiprows=1, usecols=[0], names=["raw"], encoding='utf8')
        df = predict_all(df)
        df.to_csv("export.csv", index=False, sep=",", encoding='utf-8-sig')
        return render_template("download.html", file_csv= "Download")

    elif fileExtension == '.xlsx' or fileExtension == '.xls':
        xls = pd.ExcelFile(f)
        writer = pd.ExcelWriter('export.xlsx', engine='xlsxwriter')
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(f, skiprows=1, usecols=[0], names=["raw"], sheet_name=sheet_name, encoding='utf8')
            df = predict_all(df)
            df.to_excel(writer, sheet_name=sheet_name, index=False, sep=",", encoding='utf-8-sig')
        writer.save()
        return render_template("download.html", file_xsl="Download")

    else:
        content= "Incorrect file format. Please try again!"
        return render_template('predict.html', content=content)

@app.route('/download_csv')
def download_file():
    return send_file("export.csv", as_attachment=True,
                     attachment_filename='export.csv',
                     mimetype='text/csv')

@app.route('/download_xsl')
def download_file_2():
    return send_file("export.xlsx", as_attachment=True,
                     attachment_filename='export.xlsx',
                     mimetype='application/vnd.ms-excel')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port, debug=True)