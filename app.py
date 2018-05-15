import pickle
import random

from flask import Flask, request, jsonify
from flask_cors import CORS

from model.sample_generator import create_sample
from config import all_args, kondzilla_args, proibidao_args, ostentacao_args


app = Flask(__name__)
CORS(app)


# load the model
def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


all_songs = load('generated_songs/generated-all-songs.pkl')
kondzilla_songs = load('generated_songs/generated-kondzilla-songs.pkl')
proibidao_songs = load('generated_songs/generated-proibidao-songs.pkl')
ostentacao_songs = load('generated_songs/generated-ostentacao-songs.pkl')

all_sampler = create_sample(all_args)
kondzilla_sampler = create_sample(kondzilla_args)
proibidao_sampler = create_sample(proibidao_args)
ostentacao_sampler = create_sample(ostentacao_args)


def get_song(model_id):
    random_num = random.randint(0, len(all_songs) - 1)

    if model_id == 1:
        return all_songs[random_num][1:].replace('\n', '<br>')
    elif model_id == 2:
        return kondzilla_songs[random_num][1:].replace('\n', '<br>')
    elif model_id == 3:
        return proibidao_songs[random_num][1:].replace('\n', '<br>')
    else:
        return ostentacao_songs[random_num][1:].replace('\n', '<br>')


# API route
@app.route('/api', methods=['POST'])
def api():
    """API function

    All model-specific logic to be defined in the get_model_api()
    function
    """
    data = request.json
    model_id = int(data['id'])
    prime_words = data['sentence']

    if prime_words == '':
        output_data = get_song(model_id)
    else:
        output_data = -1
        if model_id == 1:
            while output_data == -1:
                output_data = all_sampler(prime_words, html=True)
        elif model_id == 2:
            while output_data == -1:
                output_data = kondzilla_sampler(prime_words, html=True)
        elif model_id == 3:
            while output_data == -1:
                output_data = proibidao_sampler(prime_words, html=True)
        else:
            while output_data == -1:
                output_data = ostentacao_sampler(prime_words, html=True)

    return jsonify(output_data)


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


if __name__ == '__main__':
    # This is used when running locally.
    app.run(host='0.0.0.0', debug=True)
