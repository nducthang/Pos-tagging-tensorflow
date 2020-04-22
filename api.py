from flask import Flask, render_template, request, url_for, jsonify
import config.config as config
from utils import load_config
import tensorflow as tf
from train.model_tensorflow.model_tensorflow import Model
from flask_cors import CORS  # pip install -U flask-cors
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

app = Flask(__name__)
CORS(app)

LOAD_CHECKPOINT = config.LOAD_CHECKPOINT
LOAD_ALLWORDS = config.LOAD_ALLWORDS
LOAD_ALLTAGS = config.LOAD_ALLTAGS
LOAD_WORD2IDX = config.LOAD_WORD2IDX
LOAD_IDX2WORD = config.LOAD_IDX2WORD
LOAD_TAG2IDX = config.LOAD_TAG2IDX
LOAD_IDX2TAG = config.LOAD_IDX2TAG

# Load model
all_words, all_tags, word2idx, idx2word, tag2idx, idx2tag = load_config(LOAD_ALLWORDS, LOAD_ALLTAGS, LOAD_WORD2IDX,
                                                                        LOAD_IDX2WORD, LOAD_TAG2IDX, LOAD_IDX2TAG)
VOCAB_SIZE = len(word2idx.items())
NUM_CLASSES = len(tag2idx.items())


def get_model_api():
    model = Model(VOCAB_SIZE, NUM_CLASSES)
    model.build_model()
    model.saver.restore(model.sess, tf.train.latest_checkpoint(LOAD_CHECKPOINT))

    def model_api(input_data):
        sent_token, tags = model.predict_batch(input_data, all_words, word2idx, idx2tag)
        output_data = {'tokens': sent_token, 'tags': tags}
        return output_data

    return model_api


model_api = get_model_api()


@app.route('/')
def index():
    return "Index API"


# HTTP Errors handlers
@app.errorhandler(404)
def url_error(e):
    return """
    Wrong URL!
    <pre>{}</pre>""".format(e), 404


@app.errorhandler(500)
def server_error(e):
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500


@app.route('/api', methods=['POST', 'OPTIONS'])
def predict():
    api_input = request.json
    if api_input is not None:
        sentence = [str(api_input["text"])]
        response = model_api(sentence)
        return jsonify(response)
    else:
        return "Don't send None data to server"


if __name__ == '__main__':
    app.run(debug=False)
