from typing import Optional
from fastapi import FastAPI
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk
from nltk import word_tokenize
from flair.data import Sentence
from flair.models import SequenceTagger
import numpy as npd
import math
from difflib import SequenceMatcher
from collections import OrderedDict


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def check_pos_tagging(a, pos_tagging):
    for word in pos_tagging:
        if word[0] == a:
            # print(word[1])
            return word[1]


app = FastAPI()


@app.get("/extractor/{content}")
def text_rank(content):
    # 1. case folding
    text = content.lower()
    text = word_tokenize(text)
    plain = [character for character in text if character.isalnum()]
    text = " ".join(plain)

    # 2. stemming
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    text = stemmer.stem(text)

    # 3. tokenizing
    text_tokenizing = word_tokenize(text)
    # print(" ".join(text_tokenizing))

    # 5. pos-tagging
    sentence = Sentence(text)
    tag_pos = SequenceTagger.load('/content/gdrive/MyDrive/skripsi-asif/tagger/best-model.pt')
    tag_pos.predict(sentence)
    tagging = sentence.to_tagged_string().split(" ")
    pos_tagging = []
    for i in range(len(text_tokenizing)):
        pos_tagging.append([text_tokenizing[i], tagging[i * 2 + 1]])

    # 4. filtering
    stopwords = []
    wanted_pos = ['<NOUN>', '<ADJ>', '<VERB>', '<X>', '<PROPN>']
    for word in pos_tagging:
        if word[1] not in wanted_pos:
            stopwords.append(word[0])
    stopwords_file = open("/content/gdrive/MyDrive/skripsi-asif/tagger/stopwords.txt", "r")
    lots_of_stopwords = []
    for line in stopwords_file.readlines():
        lots_of_stopwords.append(str(line.strip()))
    stopwords_plus = stopwords + lots_of_stopwords
    stopwords_plus = set(stopwords_plus)
    processed_text = []
    for word in text_tokenizing:
        if word not in stopwords_plus:
            processed_text.append(word)
    vocabulary = list(set(processed_text))

    # 6. building graph
    vocab_len = len(vocabulary)
    weighted_edge = np.zeros((vocab_len, vocab_len), dtype=np.float32)

    score = np.zeros((vocab_len), dtype=np.float32)
    window_size = 3
    covered_concurrences = []

    for i in range(0, vocab_len):
        score[i] = 1
        for j in range(0, vocab_len):
            if j == i:
                weighted_edge[i][j] = 0
            else:
                for window_start in range(0, (len(processed_text) - window_size + 1)):
                    window_end = window_start + window_size
                    window = processed_text[window_start:window_end]
                    if (vocabulary[i] in window) and (vocabulary[j] in window):
                        index_of_i = window_start + window.index(vocabulary[i])
                        index_of_j = window_start + window.index(vocabulary[j])
                        if [index_of_i, index_of_j] not in covered_concurrences:
                            weighted_edge[i][j] += 1 / math.fabs(index_of_i - index_of_j)
                            covered_concurrences.append([index_of_i, index_of_j])
    inout = np.zeros(vocab_len, dtype=np.float32)

    for i in range(0, vocab_len):
        for j in range(0, vocab_len):
            inout[i] += weighted_edge[i][j]

    # 7. Scoring Vertices
    MAX_ITERATIONS = 50
    d = 0.85
    threshold = 0.0001  # convergence threshold

    for iter in range(0, MAX_ITERATIONS):
        prev_score = np.copy(score)
        for i in range(0, vocab_len):
            summation = 0
            for j in range(0, vocab_len):
                if weighted_edge[i][j] != 0:
                    summation += (weighted_edge[i][j] / inout[j]) * score[j]
            score[i] = (1 - d) + d * (summation)
        if np.sum(np.fabs(prev_score - score)) <= threshold:  # convergence condition
            break
    phrases = []

    phrase = " "
    for word in text_tokenizing:
        if word in stopwords_plus:
            if phrase != " ":
                phrases.append(str(phrase).strip().split())
            phrase = " "
        elif word not in stopwords_plus:
            phrase += str(word)
            phrase += " "

    unique_phrases = []

    for phrase in phrases:
        if phrase not in unique_phrases:
            unique_phrases.append(phrase)

    for word in vocabulary:
        for phrase in unique_phrases:
            if (word in phrase) and ([word] in unique_phrases) and (len(phrase) > 1):
                unique_phrases.remove([word])

    # 8. Scoring Keyphrases
    phrase_scores = []
    keywords = []
    for phrase in unique_phrases:
        phrase_score = 0
        keyword = ''
        for word in phrase:
            keyword += str(word)
            keyword += " "
            phrase_score += score[vocabulary.index(word)]
        phrase_scores.append(phrase_score)
        keywords.append(keyword.strip())

    # 9. Ranking Keyphrases
    sorted_index = np.flip(np.argsort(phrase_scores), 0)
    adjustment = []
    for i in range(0, 10):
        adjustment.append(str(keywords[sorted_index[i]]))

    # contents = ""
    # for i in range(0, 10):
    #     contents = contents + str(keywords[sorted_index[i]]) + "\n"

    # print("text rank by : TextRank: Bringing Order into Texts - by Rada Mihalcea and Paul Tarau")
    # print(contents)

    # 10. Indonesian adjustment
    # adjustment approach taken from Wongchaisuwat and Qingyun research
    # 10.1 - 10.5 taken from Wongchaisuwat research
    # Keyword Scores Computation after Sentences Scores Computation(1-9)

    # 10.1 Split keyword to words that often appear
    tt = " ".join(text_tokenizing)
    final = []
    for i in adjustment:
        a = i.split()
        c = a[0]
        for j in range(1, len(a)):
            if tt.count(c) == 1 and tt.count(a[j]) == 1 and check_pos_tagging(a[j], pos_tagging) == "<NOUN>":
                c += " " + a[j]
            elif tt.count(c + " " + a[j]) > 1:
                c += " " + a[j]
            else:
                final.append(c)
                c = a[j]
        final.append(c)
    for i in range(len(final)):
        final[i] = final[i].strip()
    final = list(set(final))

    # 10.2 Scoring before elimination similarity word
    sf = [len(final[i].split()) for i in range(len(final))]
    unique = []
    scores = []
    for i in range(0, len(final)):
        if sf[i] != 0:
            t = final[i].split()
            s = 0
            for k in t:
                s += score[vocabulary.index(k)]
            unique.append(final[i])
            scores.append(s)
    for i in range(0, len(scores) - 1):
        for j in range(0, len(scores) - i - 1):
            if scores[j] < scores[j + 1]:
                scores[j], scores[j + 1] = scores[j + 1], scores[j]
                unique[j], unique[j + 1] = unique[j + 1], unique[j]
    final = unique

    # 10.3 and 10.4 Qingyun research
    # 10.3 Elimination similarity word
    for i in range(0, len(final)):
        if sf[i] == 0:
            continue
        for j in range(i + 1, len(final)):
            if sf[j] == 0:
                continue
            elif final[i] in final[j]:
                sf[i] = 0
                break
            elif final[j] in final[i]:
                sf[j] = 0
            elif similar(final[i], final[j]) > 0.5:
                if len(final[i]) < len(final[j]):
                    sf[i] = 0
                    break
                else:
                    sf[j] = 0
    # 10.4 Scoring after elimination similarity word
    unique = []
    scores = []
    for i in range(0, len(final)):
        if sf[i] != 0:
            t = final[i].split()
            s = 0
            for k in t:
                s += score[vocabulary.index(k)]
            unique.append(final[i])
            scores.append(s)
    for i in range(0, len(scores) - 1):
        for j in range(0, len(scores) - i - 1):
            if scores[j] < scores[j + 1]:
                scores[j], scores[j + 1] = scores[j + 1], scores[j]
                unique[j], unique[j + 1] = unique[j + 1], unique[j]

    # 10.5 Ranking adjusment Indonesian Keyphrases
    keywords_num_0 = 5
    last = []
    for i in range(0, keywords_num_0):
        last.append(unique[i])
    final = []
    for i in last:
        v = i.split()
        temp = []
        for j in v:
            temp.append(plain[text_tokenizing.index(j)])
        final.append(" ".join(temp))

    return {"keyword": final, 'keyphrases': adjustment}


@app.get("/")
def read_root():
    return {"Hello": "World"}
