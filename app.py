#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 00:36:27 2017

@author: IbrahimM
"""

from flask import Flask
from flask import render_template
from flask import request
import ml_script_pennapps
import base64

import io
import random
import operator
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


import json

from flask_flatpages import FlatPages
from flask_frozen import Freezer

app = Flask(__name__)
app.config.from_pyfile('settings.py')
pages = FlatPages(app)
freezer = Freezer(app)
 
mp  =ml_script_pennapps.ModelPrediction()

 
@app.route('/',  methods=['POST','GET'])
def mainPage():
    return render_template('index.html')
    
@app.route('/index.htm',  methods=['POST','GET'])
def indexPage():
    return render_template('index.html')
    
@app.route('/about.htm',  methods=['POST','GET'])
def aboutPage():
    return render_template('about.htm')

@app.route('/contact.htm',  methods=['POST','GET'])
def contactPage():
    return render_template('contact.htm')
    
@app.route('/chartDisplay', methods=['POST','GET'])
def chartDisplay():
    my_text = request.form["username"]
    dump= json.dumps({"text" : my_text})
    text = json.loads(dump).get("text")
    scores = mp.predict_on_text(text)[0]
    dic = {}
    dic["disgust"] = scores[0]*100
    dic["shame"] = scores[1]*100
    dic["sadness"] = scores[2]*100
    dic["anger"] = scores[3]*100
    dic["fear"] = scores[4]*100
    dic["joy"] = scores[5]*100
    dic["guilt"] = scores[6]*100

    plt.figure(1)
    plt.xticks(np.arange(0, 110, 10))
    y = []
    labels = []
    i = 0.5
    sorted_x = sorted(dic.items(), key=operator.itemgetter(1))
    for key, value in sorted_x: 
        ax = plt.gcf().gca()
        ax.add_patch(patches.Rectangle((0,i-.25), value, 0.5, facecolor=getColor(key)))
        y.append(i)
        labels.append(key)
        i += 1.0
    y.append(i)
    plt.xlabel('Probabilities')
    plt.ylabel('Sentiments/Emotions')
    plt.yticks(y)
    ax.set_yticklabels(labels, rotation='horizontal', fontsize=12)
    plt.suptitle('Likelihood of Each Emotion')

    img = io.BytesIO()
    plt.savefig(img, format = 'png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return '<img src="data:image/png;base64,{}">'.format(plot_url)
    
def getColor(s):
    color_d = {'disgust':'green', 
          'shame':'purple',
          'sadness':'blue',
          'anger':'red',
          'fear':'orange',
          'joy':'yellow',
          'guilt':'black'}
    color_ls = ['green', 'purple', 'blue', 'red', 'orange', 'yellow', 'black', 'pink', 'gray']
    if s in color_d:
        return color_d[s]
    else:
        return random.choice(color_ls)
 
if __name__ == "__main__":
    app.run(debug = True)
