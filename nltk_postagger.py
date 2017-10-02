# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 12:04:02 2017

@author: user
"""

import nltk
from nltk.corpus import brown

text1 = nltk.word_tokenize("I have been to Hawaii twice with my family")
print(nltk.pos_tag(text1))
print()

text2 = nltk.word_tokenize("She leaves the house to get the leaves")
print(nltk.pos_tag(text2))
print()

print(brown.tagged_words())
print()

brown_tagged_sents = brown.tagged_sents(categories='news')
brown_sents = brown.sents(categories='news')

# Cara 1
default_tagger = nltk.DefaultTagger('NN')
eval = default_tagger.evaluate(brown_tagged_sents)
print("Akurasi default tagger:")
print(eval)
print()