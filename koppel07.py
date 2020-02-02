#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Robert Paßmann
# hw56@iu.edu
# BEST CONFIGURATIONS in Koppel07:
# PAN12I, ELIMINATE = 3, ITERATIONS = 10

# pwd = '/Users/hwang/Desktop/repo/Unmasking-XSYYZ'

import logging
import argparse
import jsonhandler
import sys
from sklearn import svm
import numpy as np
from collections import Counter
import random
import re

# number of features to be eliminated from each extreme (max/min +/-)
FOLD_ELIMINATION = 3
NUMBER_ITERATIONS = 10
CHUNK_LENGTH = 500  # 考虑到原文是在英文基础上做的，中英文互译比例，中:英=1.5-2:1，这里chunk的长度应该更长一些
INITIAL_FEATURE_SET_LENGTH_UNIGRAM = 100
INITIAL_FEATURE_SET_LENGTH_BIGRAM = 150


def sent_tokenizer_ch(text):
    """
    这个有点问题，譬如会把下面这个带冒号的完整句子断开
    又曰：“十方世界，一切诸佛，知诸众生，乐欲不同，随其所应，说法调服。”
    """
    import re
    text = text.replace('\n', "")
    text = text.replace(' ', "")
    resentencesp = re.compile('([﹖﹗；。！？]["’”」』]{0,2})')

    s = text
    slist = []
    for i in resentencesp.split(s):
        if resentencesp.match(i) and slist:
            slist[-1] += i
        elif i:
            slist.append(i)
    return slist


def char_tokenizer_ch(sent_tokens):
    noPunckString = ""
    for sent_token in sent_tokens:
        sent_token = [re.sub("""[\s+\.\!\/_,$%^*(+\"\']+|[；：+——！，。？、~@#￥%……&*（）<>《》“”]+""", "", sent_token)]
        noPunckString += sent_token[0]
    noPuncCharTokens = [char for char in noPunckString]

    return noPuncCharTokens


def chunk2tokens(chunk):
    tokens = []
    for sent in sent_tokenizer_ch(chunk):
        tokens.extend(char_tokenizer_ch(sent))
    return tokens


def select_chunks(text1, text2):
    """
    Reduce the number of chunks of the text with more chunks such that
    we have the same number of chunks for both texts, i.e. randomly delete
    chunks from the text with more chunks
    """
    random.seed()
    text1.selected_chunks = text1.chunks
    text2.selected_chunks = text2.chunks
    # random.choice 随机删除一个seq中的值
    while len(text1.selected_chunks) > len(text2.selected_chunks):
        text1.selected_chunks.remove(random.choice(text1.selected_chunks))
    while len(text2.selected_chunks) > len(text1.selected_chunks):
        text2.selected_chunks.remove(random.choice(text2.selected_chunks))


def curve_score(curve):
    """Calculate a score for a curve by which they are sorted"""
    # this should be optimized
    return sum(curve)


class Database:
    """represents a database with texts of known authors"""

    def __init__(self):
        self.authors = []  # a list of strings with names of authors
        self.texts = {}  # a dictionary (name:Text) keys are cands names
        self.features = {}

    def add_author(self, *authors):
        for author in authors:
            self.authors.append(author)
            self.texts[author] = []

    def add_text(self, author, text):
        """
        Keyword arguments:
        author -- an author whose texts we want to add texts -- a list of texts of this author
        """
        (self.texts[author]).append(text)  # text是个Text类的实例，texts是database的属性

    def feature_generator(self, author):
        """
        Calculate the initial feature set consisting of the most frequent
        INITIAL_FEATURE_SET_LENGTH (250) words
        for every text chunks have to be created beforehand
        feature set这里做tokens讲
        """
        self.features[author] = []

        counter_unigram = Counter((self.texts[author][0]).unigram)
        counter_bigram = Counter((self.texts[author][0]).bigram)
        self.features[author].extend(list(dict(counter_unigram.most_common(INITIAL_FEATURE_SET_LENGTH_UNIGRAM)).keys()))
        self.features[author].extend(list(dict(counter_bigram.most_common(INITIAL_FEATURE_SET_LENGTH_BIGRAM)).keys()))


class Text:
    """represents a text"""

    def __init__(self, raw, name):
        """
        Keyword arguments:
        raw -- The raw text as a string.
        name -- The name of the text.
        """
        self.raw = raw
        self.name = name
        self.chunks = []  # containing all the chunks of n words, a list of lists
        self.selected_chunks = []  # contains a reduced number of chunks
        self.tokens = []
        self.sent_tokens = []
        self.unigram = []  # is self.tokens
        self.bigram = []

    def ngrams_generator(self):
        """
        This function should only be used after "chunks_generator" function.
        """

        def find_bigrams(input_list):
            bigram_list = []
            for i in range(len(input_list) - 1):
                bigram_list.append((str(input_list[i]) + str(input_list[i + 1])))
            return bigram_list

        self.unigram = self.tokens
        self.bigram = find_bigrams(self.tokens)

    def chunks_generator(self):
        global CHUNK_LENGTH
        self.sent_tokens = sent_tokenizer_ch(self.raw)
        self.tokens = char_tokenizer_ch(self.sent_tokens)

        ceiling = len(self.sent_tokens)
        i = 1  # "i-1" denotes the index of the first sent_tokens of chunks
        word_count = 0  # "word_count" used to count how many word in a chunk without breaking up sentence structure
        chunk = ''  # "chunk" used to save sentence tokens

        while True:
            for sent_token in self.sent_tokens[i - 1:]:
                i += 1
                if word_count <= 500:
                    chunk += sent_token
                    word_count += len(char_tokenizer_ch(sent_token))
                elif i == ceiling:
                    break
                else:
                    self.chunks.append([chunk])
                    # print('a')
                    word_count = 0
                    chunk = ''
            break


def heavy_lifter(corpusdir, outputdir):
    # load training data
    jsonhandler.loadJson(corpusdir)
    # 现在吃进meta-file.json: corpusdir, upath, candidates, unknowns, encoding, language
    # corpusdir是meta-file.json所在文件夹 ,upath是"unknown"文件夹的位置，也就是放unknown texts的地方
    # unknowns是unknowns-texts的文件名，ie. "unknown00011.txt"
    jsonhandler.loadTraining()
    # 现在trainings（dict）有内容了，ie:{...'candidate00010': ['known00001.txt', 'known00002.txt']}
    database = Database()   # database.authors = [], database.texts = {}, database.features = {}

    for candidate in jsonhandler.candidates:
        database.add_author(candidate)  # database.authors = [cand1, cand2...]// database.texts = {author:[]...}
        for training in jsonhandler.trainings[candidate]:  # 对于某个candidate的training text的文件名
            logging.info("Reading training text '%s' of '%s'", training, candidate)
            text = Text(jsonhandler.getTrainingText(candidate, training),  # 创建Text实例 text(raw, name)
                        candidate + " " + training)  # name类似于 "candidate00001 known00002.txt"
            try:
                text.chunks_generator()  # Text.chunks是一个list of lists（由许多chunks组成，每个chunk里是tokens列表）
                text.ngrams_generator()  # 生成text实例的ngram
                database.add_text(candidate, text)  # 为database实例加上texts属性 {candidate00001: [text, text,...]}
                database.feature_generator(candidate)

            except:
                # logging.info("Text size too small. Skip this text.")
                logging.warning("Text too small. Exit.")
                sys.exit()

    # We use the unmasking procedure to compare all unknown texts to all
    # enumerated texts of known authorship and then decide which fit best.
    for unknown in jsonhandler.unknowns:
        results = {}  # dictionary containing the average ten folds accuracy score of each candidates
        dropped_features = {}
        # load the unknown text and create the chunks which are used for the unmasking process
        unknown_text = Text(jsonhandler.getUnknownText(unknown), unknown)
        unknown_text.chunks_generator()  # 得到unknown_text.tokens (list) 和 .chunks (list of lists)

        for candidate in jsonhandler.candidates:
            results[candidate] = []
            dropped_features[candidate] = {}
            features = (database.features[candidate]).copy()
            for known_text in database.texts[candidate]:
                experiment_scores = []
                # randomly select equally many chunks from each text
                # 重复5次实验，每次都随机删掉多余的chunk
                for experiment in range(0, 5):
                    select_chunks(unknown_text, known_text)
                    (dropped_features[candidate])[experiment] = []
                    # create label vector
                    # (0 for chunks of unknown texts, 1 for chunks of known texts)
                    label = [0 for i in range(0, len(unknown_text.selected_chunks))] + [
                        1 for i in range(0, len(known_text.selected_chunks))]
                    label = np.array(label)
                    # the reshape is necessary for the classifier
                    label.reshape(len(unknown_text.selected_chunks) + len(known_text.selected_chunks), 1)

                    # loop
                    global NUMBER_ITERATIONS
                    global FOLD_ELIMINATION
                    fold_scores = []

                    for i in range(0, NUMBER_ITERATIONS):
                        logging.info("Iteration #%s for texts '%s' and '%s'",
                                     str(i + 1), unknown, known_text.name)
                        # 算250个高频词的词频，这里除以500是因为它没有原文处理，实际要更复杂些
                        # matrix是ndarray: len(selected_chunks)*2行 * 250列 （sample个行，feature个列）
                        matrix = [[chunk[0].count(token) / len(chunk2tokens(chunk[0])) for token in features]
                                  for chunk in (unknown_text.selected_chunks + known_text.selected_chunks)]
                        matrix = np.array(matrix)

                        classifier = svm.LinearSVC()
                        classifier.fit(matrix, label)
                        fold_scores.append(classifier.score(matrix, label))

                        # a list of all feature weights
                        weights = classifier.coef_[0]

                        # Now, we have to delete the strongest weighted features from each side.
                        # indices of maximum/minimum ROUND_ELIMINATION values
                        delete = list(np.argsort(weights)[-FOLD_ELIMINATION:]) \
                                       + list(np.argsort(weights)[:FOLD_ELIMINATION])

                        # We cannot directly use the delete list to eliminate from
                        # the features list since peu-a-peu elimination changes the indices.
                        delete_features = []
                        for x in delete:
                            delete_features.append(features[x])
                        ((dropped_features[candidate])[experiment]).append(delete_features)
                        logging.info("Delete %s", str(delete_features))

                        for feature in delete_features:
                            if feature in features:
                                features.remove(feature)
                    experiment_scores.append(fold_scores)

                experiment_scores = np.array(experiment_scores)
                experiment_scores = np.mean(experiment_scores, axis=0)
                results[candidate] = experiment_scores

    # save everything in the specified directory
    jsonhandler.storeJson(outputdir, results, dropped_features)


def main():
    parser = argparse.ArgumentParser(description="Let's find out who is the real author of XSYYL now.")
    parser.add_argument('-i',
                        action='store',
                        help='Path to input directory')
    parser.add_argument('-o',
                        action='store',
                        help='Path to output directory')

    args = vars(parser.parse_args())

    corpusdir = args['i']
    outputdir = args['o']

    heavy_lifter(corpusdir, outputdir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING,
                        format='%(asctime)s %(levelname)s: %(message)s')
    main()
