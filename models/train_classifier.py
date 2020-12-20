#import libraries
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
from sqlalchemy import create_engine
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
import nltk
import re
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline , FeatureUnion
from nltk.corpus import stopwords
from nltk import pos_tag , ne_chunk
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import pickle


def load_data(database_filepath):

    '''
    loading cleaned table created and loaded in .db in previous section.

    INPUT: database_filepath (note that your data in saved in separate folder and not in thic current directory)

    OUTPUT: features (X), labels (Y)

    '''

        #engine = create_engine('sqlite:///{}'.format(database_filepath))
        #df = pd.read_sql("SELECT * FROM disaster_response", engine)
    db = sqlite3.connect(database_filepath)
    cursor = db.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()[0][0]
    df = pd.read_sql_query('SELECT * FROM '+tables,db)

    #print(df)



    #just to make sure the function is working properly
    print(df.head())
    X = df['message']
    Y = df.drop(['id','message','original','genre'],axis=1)
    category_names=list(Y.columns)
    print(X)
    print(Y)
    print(category_names)
    return X,Y,category_names


url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
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



def build_model():
    '''
    implement ML pipeline and optimize parameters using Grid Search Algorithms


    OUTPUT: The model and the optimized parameters
    '''
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('match_word', MatchWord())
        ])),

        ('clf',MultiOutputClassifier(KNeighborsClassifier()))
    ])
    parameters = {
        'clf__estimator__n_neighbors': [5,10],
        #'clf__estimator__weights' : ['uniform', 'distance'],
        #'tfidf__smooth_idf': (True, False)
    }
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv







def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluate the f1score, precision and recall for each output category
    using `classification_report`.

    INPUT: the model created in previous function, X_test (features) Y_test(labels)
    and category_names created in load_data() function

    OUTPUT:
    summary of evaluation for each column separetely
    '''

    Y_pred = model.predict(X_test)
    Y_pred_df=pd.DataFrame(Y_pred,columns=category_names)
    #Y_test_df=pd.DataFrame(Y_test,columns=category_names)
    for col in list(Y_test.columns):
        print('------------------------------------------------')
        print('this is evaluation result for {}'.format(col))
        print(classification_report(Y_test[col], Y_pred_df[col]))
        print('------------------------------------------------')



def save_model(model, model_filepath):
    #joblib.dump(model.best_estimator_, model_filepath)
    pickle.dump(model,open(model_filepath,'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
