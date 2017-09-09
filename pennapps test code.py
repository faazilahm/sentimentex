# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 00:28:33 2017

@author: Saniyah
"""
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def getColor(i):
    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink', 'black', 'gray']
    i = i % len(colors)
    return colors[i]

# dict is a dictionary of emotions to integer plotting values
def plot_emotions(dic, save):
    plt.figure(1)
    x = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]
    plt.yticks(np.arange(0, 110, 10))
    for i in range(len(x)-1):    
        ax = plt.gcf().gca()
        ax.add_patch(patches.Rectangle((i+0.25,0), 0.5, dic[int(i)], facecolor=getColor(i)))
    plt.ylabel('Percentages')
    plt.xlabel('Emotions')
    for elt in x:
        elt = elt * 50
    plt.xticks(x)
    plt.suptitle('Percentage of Each Emotion')
    plt.savefig(save + '.png', bbox_inches='tight')
    plt.show()
    
rand = {}
for x in range(8):
    num = random.choice(range(100))
    rand.update({x:num})
print (rand)
plot_emotions(rand, 'testing')