# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 00:28:33 2017

@author: Saniyah
"""
import random, operator
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

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

# dict is a dictionary of emotions to integer plotting values
def plot_emotions(dic, save):
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
    plt.xlabel('Percentages')
    plt.ylabel('Emotions')
    plt.yticks(y)
    ax.set_yticklabels(labels, rotation='horizontal', fontsize=12)
    plt.suptitle('Percentage of Each Emotion')
    plt.savefig(save + '.png', bbox_inches='tight')
    plt.show()
    
rand = {}
y_ticks_labels = ['disgust', 'shame', 'sadness', 'anger', 'fear', 'joy', 'guilt']
for x in range(7):
    num = random.choice(range(100))
    rand.update({y_ticks_labels[x]:num})
print (rand)
plot_emotions(rand, 'testing')