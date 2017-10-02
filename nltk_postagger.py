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
size = int(len(brown_tagged_sents) * 0.9)

# Cara 1
default_tagger = nltk.DefaultTagger('NN')
eval = default_tagger.evaluate(brown_tagged_sents)
print("Akurasi default tagger:")
print(eval)
print()

# Cara 2: regex tagger
patterns = [
    (r'.*ing$', 'VBG'),               # gerunds
    (r'.*ed$', 'VBD'),                # simple past
    (r'.*es$', 'VBZ'),                # 3rd singular present
    (r'.*ould$', 'MD'),               # modals
    (r'.*\'s$', 'NN$'),               # possessive nouns
    (r'.*s$', 'NNS'),                 # plural nouns
    (r'^-?[0-9]+(.[0-9]+)?$', 'CD'),  # cardinal numbers
    (r'.*', 'NN')                     # nouns (default)
]
regex_tagger = nltk.RegexpTagger(patterns)
eval2 = regex_tagger.evaluate(brown_tagged_sents)
print("Akurasi regex tagger:")
print(eval2)
print()

fd = nltk.FreqDist(brown.words(categories='news'))
cfd = nltk.ConditionalFreqDist(brown.tagged_words(categories='news'))
most_freq_words = fd.most_common(100)
likely_tags = dict((word, cfd[word].max()) for (word, _) in most_freq_words)
baseline_tagger = nltk.UnigramTagger(model=likely_tags)
eval3 = baseline_tagger.evaluate(brown_tagged_sents[size:])
print("Akurasi lookup tagger (size=100):")
print(eval3)
print()

most_freq_words = fd.most_common(500)
likely_tags = dict((word, cfd[word].max()) for (word, _) in most_freq_words)
baseline_tagger = nltk.UnigramTagger(model=likely_tags)
eval4 = baseline_tagger.evaluate(brown_tagged_sents[size:])
print("Akurasi lookup tagger (size=500):")
print(eval4)
print()

most_freq_words = fd.most_common(1000)
likely_tags = dict((word, cfd[word].max()) for (word, _) in most_freq_words)
baseline_tagger = nltk.UnigramTagger(model=likely_tags)
eval5 = baseline_tagger.evaluate(brown_tagged_sents[size:])
print("Akurasi lookup tagger (size=1000):")
print(eval5)
print()

baseline_tagger = nltk.UnigramTagger(model=likely_tags, backoff=nltk.DefaultTagger('NN'))
eval6 = baseline_tagger.evaluate(brown_tagged_sents[size:])
print("Akurasi lookup tagger (size=1000) + backoff tagger:")
print(eval6)
print()

train_sents = brown_tagged_sents[:size]
test_sents = brown_tagged_sents[size:]
unigram_tagger = nltk.UnigramTagger(train_sents)
eval7 = unigram_tagger.evaluate(test_sents)
print("Akurasi unigram tagger:")
print(eval7)
print()

bigram_tagger = nltk.BigramTagger(train_sents)
eval8 = bigram_tagger.evaluate(test_sents)
print("Akurasi bigram tagger:")
print(eval8)
print()

t0 = nltk.DefaultTagger('NN')
t1 = nltk.UnigramTagger(train_sents, backoff=t0)
t2 = nltk.BigramTagger(train_sents, backoff=t1)
eval9 = t2.evaluate(test_sents)
print("Akurasi combined tagger:")
print(eval9)
print()