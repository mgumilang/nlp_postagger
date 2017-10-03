import nltk
from nltk.corpus import brown

brown_tagged_sents = brown.tagged_sents(categories='news')
size = int(len(brown_tagged_sents) * 0.9)
train_sents = brown_tagged_sents[:size]
test_sents = brown_tagged_sents[size:]
brown_tags_words = [ ]
for sent in train_sents:
    brown_tags_words.append( ("START", "START") )
    brown_tags_words.extend([ (tag[:2], word) for (word, tag) in sent ])
    brown_tags_words.append( ("END", "END") )

# conditional frequency distribution
cfd_tagwords = nltk.ConditionalFreqDist(brown_tags_words)
# conditional probability distribution
cpd_tagwords = nltk.ConditionalProbDist(cfd_tagwords, nltk.MLEProbDist)

# Estimating P(ti | t{i-1}) from corpus data using Maximum Likelihood Estimation (MLE):
# P(ti | t{i-1}) = count(t{i-1}, ti) / count(t{i-1})
brown_tags = [tag for (tag, word) in brown_tags_words ]

cfd_tags= nltk.ConditionalFreqDist(nltk.bigrams(brown_tags))
cpd_tags = nltk.ConditionalProbDist(cfd_tags, nltk.MLEProbDist)

distinct_tags = set(brown_tags)

def classify(sentence):
    viterbi = [ ]
    backpointer = [ ]
    
    first_viterbi = { }
    first_backpointer = { }
    for tag in distinct_tags:
        # don't record anything for the START tag
        if tag == "START": continue
        first_viterbi[ tag ] = cpd_tags["START"].prob(tag) * cpd_tagwords[tag].prob( sentence[0] )
        first_backpointer[ tag ] = "START"
        
    viterbi.append(first_viterbi)
    backpointer.append(first_backpointer)
    
    for wordindex in range(1, len(sentence)):
        this_viterbi = { }
        this_backpointer = { }
        prev_viterbi = viterbi[-1]
        
        for tag in distinct_tags:
            # don't record anything for the START tag
            if tag == "START": continue
    
            best_previous = max(prev_viterbi.keys(),
                                key = lambda prevtag: \
                prev_viterbi[ prevtag ] * cpd_tags[prevtag].prob(tag) * cpd_tagwords[tag].prob(sentence[wordindex]))
    
            this_viterbi[ tag ] = prev_viterbi[ best_previous] * \
                cpd_tags[ best_previous ].prob(tag) * cpd_tagwords[ tag].prob(sentence[wordindex])
            this_backpointer[ tag ] = best_previous
    
        # done with all tags in this iteration
        # so store the current viterbi step
        viterbi.append(this_viterbi)
        backpointer.append(this_backpointer)
    
    prev_viterbi = viterbi[-1]
    best_previous = max(prev_viterbi.keys(),
                        key = lambda prevtag: prev_viterbi[ prevtag ] * cpd_tags[prevtag].prob("END"))
    
    # best tagsequence: we store this in reverse for now, will invert later
    best_tagsequence = [ "END", best_previous ]
    # invert the list of backpointers
    backpointer.reverse()
    
    # go backwards through the list of backpointers
    # (or in this case forward, because we have inverter the backpointer list)
    # in each case:
    # the following best tag is the one listed under
    # the backpointer for the current best tag
    current_best_tag = best_previous
    for bp in backpointer:
        best_tagsequence.append(bp[current_best_tag])
        current_best_tag = bp[current_best_tag]
    
    best_tagsequence.reverse()
    return best_tagsequence[1:-1]

def count_hit(list1, list2):
    counter = 0
    for i in range(len(list1)):
        if (list1[i]==list2[i]): counter += 1
        
    return counter

# Evaluation
sum_hit = 0.0
count_tag = 0
for sent in test_sents:
    test_sent = [word for (word,_) in sent]
    test_label = [tag[:2] for (_,tag) in sent]
    pred = classify(test_sent)
    sum_hit += count_hit(pred, test_label)
    count_tag += len(test_label) 

print("Train size =", len(train_sents))
print("Test size  =", len(test_sents))

acc = sum_hit / count_tag
print("Akurasi =", acc)
print()

in_user = input("Masukkan kalimat yang ingin di POS-tag:\n")
in_user = nltk.word_tokenize(in_user)
res = classify(in_user)
print(res)