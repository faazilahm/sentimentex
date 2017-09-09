# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 14:50:02 2017

@author: Saniyah
"""
import string
import pandas as pd


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
print(rate_sentence('This is a sentence with excitement!', emotions))