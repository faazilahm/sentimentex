# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 12:57:13 2017

@author: Saniyah
"""

# parse html site to get synonyms for different feelings

import urllib.request, urllib.error, urllib.parse
from bs4 import BeautifulSoup
import pandas as pd

def parse_synonyms(page, cutoff):
    dic = {} # maps synonym to rating
    # specify the url
    quote_page = page # 'https://www2.powerthesaurus.org/disgusting/synonyms'
    # query the website and return the html to the variable ‘page’
    page = urllib.request.urlopen(quote_page)
    # parse the html using beautiful soap and store in variable `soup`
    soup = BeautifulSoup(page, 'html.parser')
    # Take out the <div> of name and get its value
    word_tiles = soup.find_all("div", class_="pt-thesaurus-card")
    
    for tile in word_tiles:
        rating = int(tile.find(class_="pt-thesaurus-card__rating-count").get_text())
        if rating > cutoff:
            word = tile.find(class_="pt-thesaurus-card__term-title").a.get_text()
            dic.update({word:rating})
    
    return dic

d = parse_synonyms('https://www2.powerthesaurus.org/surprise/synonyms', 5)
for x in range(20):
    page = parse_synonyms('https://www2.powerthesaurus.org/surprise/synonyms' + '/' + str(x+1), 5)
    d.update(page)
    
raw_data = {}
words = []
ratings = []
for word, rating  in d.items():
    words.append(word)
    ratings.append(rating)
raw_data.update({"Words":words})
raw_data.update({"Ratings":ratings})
df = pd.DataFrame(raw_data, columns = ["Words", "Ratings"])
df.to_csv('Surprise.csv')
print(d)