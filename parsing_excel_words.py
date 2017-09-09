# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 11:20:06 2017

@author: Saniyah
"""

import pandas as pd

def parse_file(filepath):
    emotion_words = pd.read_csv(filepath)
    emotion_words.columns = ["Word","Emotion"]
    emotion_dict = {'disgust':set(), 
                    'shame':set(),
                    'sadness':set(),
                    'anger':set(),
                    'fear':set(),
                    'joy':set(),
                    'guilt':set(),
                    'surprise':set()}
    for _, row in emotion_words.iterrows():
        word = row['Word'] 
        emotion = row['Emotion']
        s = emotion_dict[emotion]
        s.add(word)
    
    max_size = 0
    for emotion in emotion_dict:
        size = len(emotion_dict[emotion])
        print (emotion + ': ' + str(size))
        if size > max_size:
            max_size = size
            
    # save as csv
    raw_data = {}
    cols = []
    for emotion in emotion_dict:
        ls = list(emotion_dict[emotion])
        while len(ls) < max_size:
            ls.append('')
        raw_data.update({emotion:ls})
        cols.append(emotion)
    df = pd.DataFrame(raw_data, columns = cols)
    df.to_csv('ParsedEmotionWords.csv')
  
def read_words(filepath):
    emotion_words = pd.read_csv(filepath)
    emotion_words.columns = ["Num","Disgust", "Shame", "Sadness", "Anger", 
                             "Fear", "Joy", "Guilt", "Surprise"]
    return emotion_words
    
parse_file('EmotionWords.csv')