import nltk
import sys
import re

file = open("id-ud-train.conllu", 'r', encoding="Latin1")
idn_tagged_sents = []
idn_tagged_words = []
words_idn = []
prev = 0
sents = []
for line in file:
	m = re.search('(\d+)\s(\w*)\s\w*\s(\w*)\s\w\s\w*=*\w*\s\w*\s\w*\s\w*\s\w*=*\w*', line)
	if m:
		if int(m.group(1)) <= int(prev):
			idn_tagged_sents.append(sents)
			sents = []
		idn_tagged_words.append( (m.group(2), m.group(3)) )
		sents.append( (m.group(2), m.group(3)) )
		words_idn.append(m.group(2))
		prev = m.group(1)

idn_tagged_sents_test = []
prev = 0
sents = []
file1 = open("id-ud-dev.conllu", 'r', encoding="Latin1")
for line in file1:
	m = re.search('(\d+)\s(\w*)\s\w*\s(\w*)\s\w\s\w*=*\w*\s\w*\s\w*\s\w*\s\w*=*\w*', line)
	if m:
		if int(m.group(1)) <= int(prev):
			idn_tagged_sents_test.append(sents)
			sents = []
		sents.append( (m.group(2), m.group(3)) )
		prev = m.group(1)

# Cara 1
default_tagger = nltk.DefaultTagger('NOUN')
eval = default_tagger.evaluate(idn_tagged_sents)
text1 = nltk.word_tokenize("Saya sudah pernah pergi ke hawaii")
print(default_tagger.tag(text1))
print("Akurasi default tagger:")
print(eval)
print()


fd = nltk.FreqDist(words_idn)
cfd = nltk.ConditionalFreqDist(idn_tagged_words)
most_freq_words = fd.most_common(100)
likely_tags = dict((word, cfd[word].max()) for (word, _) in most_freq_words)
baseline_tagger = nltk.UnigramTagger(model=likely_tags)
eval3 = baseline_tagger.evaluate(idn_tagged_sents)
print("Akurasi lookup tagger (size=100):")
print(eval3)
print()

most_freq_words = fd.most_common(500)
likely_tags = dict((word, cfd[word].max()) for (word, _) in most_freq_words)
baseline_tagger = nltk.UnigramTagger(model=likely_tags)
eval4 = baseline_tagger.evaluate(idn_tagged_sents)
print("Akurasi lookup tagger (size=500):")
print(eval4)
print()

most_freq_words = fd.most_common(1000)
likely_tags = dict((word, cfd[word].max()) for (word, _) in most_freq_words)
baseline_tagger = nltk.UnigramTagger(model=likely_tags)
eval5 = baseline_tagger.evaluate(idn_tagged_sents)
print("Akurasi lookup tagger (size=1000):")
print(eval5)
print()

baseline_tagger = nltk.UnigramTagger(model=likely_tags, backoff=nltk.DefaultTagger('NOUN'))
eval6 = baseline_tagger.evaluate(idn_tagged_sents)
print("Akurasi lookup tagger (size=1000) + backoff tagger:")
print(eval6)
print()

train_sents = idn_tagged_sents
test_sents = idn_tagged_sents_test
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


baseline_tagger2 = nltk.BigramTagger(model=likely_tags, backoff=nltk.UnigramTagger(model=likely_tags, backoff=nltk.DefaultTagger('NOUN')))
eval9 = baseline_tagger2.evaluate(idn_tagged_sents)
print("Akurasi combined tagger: ")
print(eval9)
print()