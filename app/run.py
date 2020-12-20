import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)


def tokenize(text):
    '''
    Process raw text message and make data ready for feature feature_extraction.
    This function is completing PROCESSING PHASE of "NLP PIPELINE"

    INPUT: raw text messages
    OUTPUT: processed text message (ready for feature extraction)
    '''
    text=text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stopwords.words("english")]
    #tokens=[PorterStemmer().stem(w) for w in tokens]
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok =  PorterStemmer().stem((((lemmatizer.lemmatize(tok, pos='v')).lower()).strip()))
        clean_tokens.append(clean_tok)

    return clean_tokens

class MatchWord(BaseEstimator, TransformerMixin):
    def match_word(self, text):

        tokenized_text = tokenize(text)
        labels = list(Y.columns)
        for word in tokenized_text:
            if word in labels:
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_matched = pd.Series(X).apply(self.match_word)
        return pd.DataFrame(X_matched)



# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('disaster_response', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    floods_counts= df.groupby('floods').count()['message']
    floods_names = list(floods_counts.index)

    fire_counts= df.groupby('fire').count()['message']
    fire_names = list(fire_counts.index)


    cold_counts= df.groupby('cold').count()['message']
    cold_names = list(cold_counts.index)

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]






    graphs = [
        {
            'data': [
                Bar(
                    x=floods_counts,
                    y=floods_names
                )
            ],

            'layout': {
                'title': 'Distribution of Message Air_Related',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "floods"
                }
            }
        }
    ]



    graphs = [
        {
            'data': [
                Bar(
                    x=cold_counts,
                    y=cold_names
                )
            ],

            'layout': {
                'title': 'Distribution of Message cold',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "cold"
                }
            }
        }
    ]




    graphs = [
        {
            'data': [
                Bar(
                    x=fire_counts,
                    y=fire_names
                )
            ],

            'layout': {
                'title': 'Distribution of Message fire',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "fire"
                }
            }
        }
    ]
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
