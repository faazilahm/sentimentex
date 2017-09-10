# Imports the Google Cloud client library
from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types
import pandas as pd
import sklearn
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from nltk.sentiment import vader
from sklearn import svm
import string
import unicodedata
import sys

sentiment= pd.read_csv('sentiment.csv')
sentiment.columns = ["index", "word", "sentiment"]
sentiment = sentiment[["word", "sentiment"]]

# Instantiates a client
client = language.LanguageServiceClient()


def print_hello():
	return("hello world")

def get_sentiment(text):
    document = types.Document(
        content=text,
        type=enums.Document.Type.PLAIN_TEXT)

    # Detects the sentiment of the text
    sentiment = client.analyze_sentiment(document=document).document_sentiment
    return sentiment.score

def get_lemma_sentence(text):
    # Gets the syntactic information of the sentence
    document = types.Document(
        content=text,
        type=enums.Document.Type.PLAIN_TEXT)

    syntax = client.analyze_syntax(document = document)
    lemmas = []
    for token in syntax.tokens:
        lemmas+= [token.lemma]
    new_sentence = " ".join(lemmas)
    return new_sentence

def get_salience_of_word(word, text):
    # Gets the syntactic information of the sentence
    document = types.Document(
        content=text,
        type=enums.Document.Type.PLAIN_TEXT)

    entity = client.analyze_entities(document = document)
    print(entity)
    for entity in entity.entities:
        name = entity.name
        if word in name:
            return entity.salience
    return 0

def get_parts_of_speech_list(text):
    # Gets the syntactic information of the sentence
    document = types.Document(
        content=text,
        type=enums.Document.Type.PLAIN_TEXT)

    syntax = client.analyze_syntax(document = document)
    tags = []
    for token in syntax.tokens:
        tags+= [token.part_of_speech.tag]
    return tags

nltk_analyzer = vader.SentimentIntensityAnalyzer()

def nltk_polarity_score_pos(text):
    return nltk_analyzer.polarity_scores(text).get("pos")

def nltk_polarity_score_neg(text):
    return nltk_analyzer.polarity_scores(text).get("neg")

def nltk_polarity_score_neutral(text):
    return nltk_analyzer.polarity_scores(text).get("neutral")

tbl = dict.fromkeys(i for i in range(sys.maxunicode)
                      if unicodedata.category(chr(i)).startswith('P'))
def remove_punctuation(text):
    return text.translate(tbl)

def get_word_distribution(text):
    text = remove_punctuation(text)
    tokenized = text.split()
    freq = [0, 0, 0, 0, 0, 0, 0]
    for word in tokenized:
        filtered = sentiment[sentiment["word"] == word]
        for emotion in filtered.sentiment:
            if emotion == "disgust":
                freq[0] +=1
            if emotion == "shame":
                freq[1]+=1
            if emotion == "sadness":
                freq[2]+=1
            if emotion == "anger":
                freq[3]+=1
            if emotion == "fear":
                freq[4]+=1
            if emotion == "joy":
                freq[5]+=1
            if emotion == "guilt":
                freq[6]+=1
    return freq


# Method 2 for Calculation

def read_words_scores(filepath):
    emotion_words = pd.read_csv(filepath)
    emotion_words.columns = ["Index", "Word", "Rating"]
    emotion_words = emotion_words[["Word","Rating"]]
    # print (emotion_words)
    return emotion_words

def set_all_emotions():
    disgust = read_words_scores('Disgust.csv')
    shame = read_words_scores('Shame.csv')
    sadness = read_words_scores('Sadness.csv')
    anger = read_words_scores('Anger.csv')
    fear = read_words_scores('Fear.csv')
    joy = read_words_scores('Joy.csv')
    guilt = read_words_scores('Guilt.csv')
    return [disgust, shame, sadness, anger, fear, joy, guilt]

def normalize(scores):
    total = 0
    for score in scores:
        total += score
    new_scores = []
    if total == 0.0:
        return scores
    for score in scores:
        new_score = float (score) / float (total)
        new_scores.append(new_score)
    return new_scores

def rate_sentence(sent, emotions):
    translator = str.maketrans('', '', string.punctuation)
    sent = sent.translate(translator)
    sent = sent.lower()
    words = sent.split(' ')
    scores = []
    for emotion in emotions:
        total = 0
        for word in words:
            filtered = emotion[emotion["Word"] == word]["Rating"]
            if not filtered.empty:
                rating = [x for x in filtered][0]
                total += rating
        scores.append(total)
    scores = normalize(scores)
    return scores

emotions = set_all_emotions()

def calculate_features_1(emotion_data):
    emotion_data.columns = ["sentiment", "sentence"]
    #emotion_data["lemma"] = [get_lemma_sentence(s) for s in emotion_data.sentence]
    #emotion_data["pos_neg"] = [get_sentiment(s) for s in emotion_data.sentence]
    #emotion_data["parts_speech"] = [get_parts_of_speech_list(s) for s in emotion_data.sentence]

    emotion_data["freq"] = [get_word_distribution(s) for s in emotion_data.sentence]
    emotion_data[['disgust','shame', "sadness", "anger", "fear", "joy", 
    "guilt"]] = pd.DataFrame([x for x in emotion_data.freq])
    return emotion_data

def calculate_features_2(emotion_data):
    emotion_data["freq_2"] = [rate_sentence(s, emotions) for s in emotion_data.sentence]
    emotion_data[['disgust_2','shame_2', "sadness_2", "anger_2", "fear_2", 
    "joy_2", "guilt_2"]] = pd.DataFrame([x for x in emotion_data.freq_2])
    return emotion_data

def random_forest_classification_test(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    clf = RandomForestClassifier(n_estimators = 100)
    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)
    print ("Accuracy: ", sklearn.metrics.accuracy_score(y_test, pred))
    print(sklearn.metrics.classification_report(y_test, pred))
    return clf

def random_forest_processing_data(emotion_data, features):
    df_X = emotion_data[features]
    df_y = emotion_data.sentiment
    predictor = random_forest_classification_test(df_X, df_y)
    return predictor
    
def SVM_predicton(emotion_data, features):
    X = emotion_data[features]
    y = emotion_data.sentiment
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    clf = svm.SVC()
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print ("Accuracy: ", sklearn.metrics.accuracy_score(y_test, pred))
    print(sklearn.metrics.classification_report(y_test, pred))
    return clf
    
def logistic_regression_prediction(emotion_data, features):
    X = emotion_data[features]
    y = emotion_data.sentiment
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    clf =  sklearn.linear_model.LogisticRegression()
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print ("Accuracy: ", sklearn.metrics.accuracy_score(y_test, pred))
    print(sklearn.metrics.classification_report(y_test, pred))
    return clf

class ModelPrediction():
    from sklearn import svm

    def __init__(self):
        self.labelled_data = pd.read_csv("labelled_emotion_data.csv")
        self.features = ["disgust", "shame", "sadness", "anger", "fear", "joy", 
        "guilt", "disgust_2", "shame_2", "sadness_2", "anger_2", "fear_2", 
        "joy_2", "guilt_2"]
        
    def train_full_model(self, X, y):
        clf = svm.SVC(probability = True)
        clf.fit(X, y)
        return clf
    def prediction_processing_data(self):
        df_X = self.labelled_data[self.features]
        df_y = self.labelled_data.sentiment
        predictor = self.train_full_model(df_X, df_y)
        return predictor

    def predict_on_text(self, text):
        predictor = self.prediction_processing_data()
        return self.predict_category(text, predictor)
        
    def predict_category(self, text, predictor):
        freq = get_word_distribution(text)
        freq_2 = rate_sentence(text, emotions)
        total_freq = freq + freq_2
        return predictor.predict_proba([total_freq])


