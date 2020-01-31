#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Robert Paßmann
# hw56@iu.edu

import logging
import argparse
import jsonhandler
import sys
from sklearn import svm
import numpy
from collections import Counter
import random
import re

# number of features to be eliminated from each extreme (max/min +/-)
NUMBER_ELIMINATE_FEATURES = 3
NUMBER_ITERATIONS = 10
# 考虑到原文是在英文基础上做的，中英文互译比例，中:英=1.5-2:1，这里chunk的长度应该更长一些
CHUNK_LENGTH = 500
INITIAL_FEATURE_SET_LENGTH = 250


# BEST CONFIGURATIONS:
# PAN12I, ELIMINATE = 3, ITERATIONS = 10
# corpusdir = "/Users/hwang/Desktop/repo/Unmasking-HLM/reference/pan12_test"
# outputdir = "/Users/hwang/Desktop/repo/Unmasking-HLM/reference/pan12_test"


def sent_tokenizer_ch(text):
    """
    这个有点问题，譬如会把下面这个带冒号的完整句子断开
    又曰：“十方世界，一切诸佛，知诸众生，乐欲不同，随其所应，说法调服。”
    """
    import re
    text = text.replace('\n', "")
    text = text.replace(' ', "")
    resentencesp = re.compile('([﹖﹗；。！？]["’”」』]{0,2})')
    # resentencesp = re.compile('([﹒﹔﹖﹗．；。！？]["’”」』]{0,2}|：(?=["‘“「『]{1,2}|$))')
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


def ngrams_tokenizer(char_tokens, n):
    """
    这里的list()可以去掉，那样直接给出一个闭包
    注意：这里实际上打破了句子的结构，一个chuck里的所有文字都是首尾相接的，这样也许不好
    """
    return list(zip(*[char_tokens[i:] for i in range(n)]))


def text_to_list(text):
    # the following takes all alphabetic words normalized to lowercase
    # from the raw data
    return [x for x in [''.join(c for c in word if c.isalpha()).lower() for word in text.split()] if x != '']


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
    # 这个与原文又有出入，原文是重复了5次实验，每次都会删掉多余的chunk
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
        self.features = []

    def add_author(self, *authors):
        for author in authors:
            self.authors.append(author)
            self.texts[author] = []

    def add_text(self, author, *texts):
        """
        Keyword arguments:
        author -- an author whose texts we want to add texts -- a list of texts of this author
        """
        if author not in self.authors:
            raise Exception("Author unknown")

        for text in texts:
            (self.texts[author]).append(text)  # text是个Text类的实例，texts是database的属性

    def feature_generator(self):
        """
        Calculate the initial feature set consisting of the most frequent
        INITIAL_FEATURE_SET_LENGTH (250) words
        for every text chunks have to be created beforehand
        feature set这里做tokens讲
        """
        counter = Counter()
        for author in self.authors:
            for text in self.texts[author]:  # 对于每一个text实例
                counter += Counter(text.tokens)  # profile-based计数
        # 这里Counter.most_common返回一个a list of tuple，外面加上dict()则变成一个dict
        # dict.keys()返回一个list
        self.feature = list(dict(counter.most_common(INITIAL_FEATURE_SET_LENGTH)).keys())


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
        # for having the same number of chunks
        # for two text during calculations
        # 之前这个tokens是词的意思，现在是中文一个字的意思，同character
        self.tokens = []
        self.sent_tokens = []

        self.chunk_feature_frequencies = {}  # 这个属性似乎没用

    # def subsample(text, scale):
    #     """
    #     Function:
    #         Subsampling a text.
    #         Require sent_tokenizer & word_tokenizer.
    #     Input:
    #         text:
    #             a str, the text need to be performed subsampling.
    #         scale:
    #             a int, denotes how many words the subsampled chunk would no less than. (Because every sentence structure
    #             should be intact.)
    #     Output:
    #         text:
    #             a str, the subsampled chunk from input.
    #     """
    #     import random
    #     subsampled_text = ''
    #     sent_tokens = sent_tokenizer(text)
    #     mark = random.randint(1, len(sent_tokens))
    #     word_token_count = 0
    #
    #     while word_token_count < scale:
    #         sub_text = sent_tokens[mark - 1]
    #         subsampled_text = subsampled_text + " " + sub_text
    #         word_token_count += len(word_tokenizer(sub_text))
    #         mark += 1
    #         if mark == len(sent_tokens):
    #             mark = 1
    #
    #     return subsampled_text

    # 重写 create_chunks—，为了1.贴合原文，二适应中文
    def create_chunks(self):
        """

        """
        global CHUNK_LENGTH
        self.sent_tokens = sent_tokenizer_ch(self.raw)
        # self.chuncks=[]在初始化时已经创建好了
        ceiling = len(self.sent_tokens)
        i = 1       # "i-1" denotes the index of the first sent_tokens of chunks
        word_count = 0      # "word_count" used to count how many word in a chunk without breaking up sentence structure
        chunk = ''      # "chunk" used to save sentence tokens

        while True:
            for sent_token in self.sent_tokens[i-1:]:
                i += 1
                if word_count <= 500:
                    chunk += sent_token
                    word_count += len(char_tokenizer_ch(sent_token))
                elif i == ceiling:
                    break
                else:
                    self.chunks.append(chunk)
                    # print('a')
                    word_count = 0
                    chunk = ''
            break


def tira(corpusdir, outputdir):
    # load training data
    jsonhandler.loadJson(corpusdir)
    # 现在吃进meta-file.json: corpusdir, upath, candidates, unknowns, encoding, language
    # corpusdir是meta-file.json所在文件夹
    # upath是"unknown"文件夹的位置，也就是放unknown texts的地方
    # unknowns是unknowns-texts的文件名，ie. "unknown00011.txt"
    jsonhandler.loadTraining()
    # 现在trainings（dict）有内容了，ie:{...'candidate00010': ['known00001.txt', 'known00002.txt']}
    # 创建数据库一个实例
    database = Database()
    # database.authors = []
    # database.texts = {}
    # database.features = []

    for candidate in jsonhandler.candidates:  # 对于每一个候选人???还可以这样？
        database.add_author(candidate)  # database.authors = [cand1, cand2...]// database.texts = {author:[]...}
        for training in jsonhandler.trainings[candidate]:  # 对于某个candidate的training text的文件名
            logging.info("Reading training text '%s' of '%s'", training, candidate)
            text = Text(jsonhandler.getTrainingText(candidate, training),  # 创建Text实例 text(raw, name)
                        candidate + " " + training)  # name类似于 "candidate00001 known00002.txt"
            try:
                text.create_chunks()  # Text.chunks是一个list of lists（由许多chunks组成，每个chunk里是许许多多的tokens）
                database.add_text(candidate, text)  # 为database实例加上texts属性 {candidate00001: [text, text,...]}
                # 此处的每个text都是一个Text类的实例
            except:
                # logging.info("Text size too small. Skip this text.")
                logging.warning("Text too small. Exit.")
                sys.exit()
    # 这里开始到特征了
    database.feature_generator()  # 找出了每个作者250个最高频的token

    candidates = []  # this list shall contain the most likely candidates

    # We use the unmasking procedure to compare all unknown texts to all
    # enumerated texts of known authorship and then decide which fit best.
    # runtime could surely be optimized
    for unknown in jsonhandler.unknowns:
        try:
            results = {}  # dictionary containing the maximum difference (first and last iteration) for every author

            # load the unknown text and create the chunks which are used for the unmasking process
            unknown_text = Text(jsonhandler.getUnknownText(unknown), unknown)
            unknown_text.create_chunks()  # 得到unknown_text.tokens (list) 和 .chunks (list of lists)

            for candidate in jsonhandler.candidates:
                results[candidate] = float("inf")

                for known_text in database.texts[candidate]:
                    # reset the feature list, i.e. create a copy of the initial list
                    features = list(database.features)  # 这个features是一个作者最常见的250个词

                    # randomly select equally many chunks from each text
                    select_chunks(unknown_text, known_text)

                    # create label vector
                    # (0 -> chunks of unknown texts, 1 -> chunks of known texts)
                    label = [0 for i in range(0, len(unknown_text.selected_chunks))] + \
                            [1 for i in range(0, len(known_text.selected_chunks))]
                    label = numpy.array(label)
                    # the reshape is necessary for the classifier
                    label.reshape(len(unknown_text.selected_chunks) + len(known_text.selected_chunks), 1)

                    # 循环开始
                    global NUMBER_ITERATIONS
                    global NUMBER_ELIMINATE_FEATURES
                    scores = []
                    for i in range(0, NUMBER_ITERATIONS):
                        logging.info("Iteration #%s for texts '%s' and '%s'",
                                     str(i + 1), unknown, known_text.name)

                    # loop
                    global NUMBER_ITERATIONS
                    global NUMBER_ELIMINATE_FEATURES
                    scores = []
                    for i in range(0, NUMBER_ITERATIONS):
                        logging.info("Iteration #%s for texts '%s' and '%s'",
                                     str(i + 1), unknown, known_text.name)
                        # Create matrix containing the relative word counts in each chunk (for the selected features)
                        # 算250个高频词的词频，这里除以500是因为它没有原文处理，实际要更复杂些
                        # matrix是ndarray: len(selected_chunks)*2行 * 250列 （sample个行，feature个列）
                        matrix = [[chunk.count(word) / CHUNK_LENGTH for word in features]
                                  for chunk in (unknown_text.selected_chunks + known_text.selected_chunks)]
                        matrix = numpy.array(matrix)

                        # Get a LinearSVC classifier and its score (i.e. accuracy in the training data).
                        # Save this score as a point in the scores curve.
                        # (We want to select the curve with the steepest decrease)
                        classifier = svm.LinearSVC()
                        classifier.fit(matrix, label)
                        scores.append(classifier.score(matrix, label))

                        # a list of all feature weights
                        flist = classifier.coef_[0]

                        # Now, we have to delete the strongest weighted features
                        # (NUMBER_ELIMINATE_FEATURES) from each side.
                        # indices of maximum 3 values and minimum 3 values
                        delete = list(numpy.argsort(flist)[-NUMBER_ELIMINATE_FEATURES:]) \
                                 + list(numpy.argsort(flist)[:NUMBER_ELIMINATE_FEATURES])

                        # We cannot directly use the delete list to eliminate from
                        # the features list since peu-a-peu elimination changes
                        # the indices.
                        delete_features = []
                        for i in delete:
                            delete_features.append(features[i])

                        logging.info("Delete %s", str(delete_features))

                        for feature in delete_features:
                            # a single feature could appear twice in the delete
                            # list
                            if feature in features:
                                features.remove(feature)

                    # The scores list is now the graph we use to get our results
                    # Therefore, compare with previous scores.
                    score = curve_score(scores)
                    logging.info("Calculated a score of %s", str(score))
                    if score < results[candidate]:
                        results[candidate] = score

            # Which author has the biggest score?
            most_likely_author = min(results, key=results.get)
            logging.info("Most likely author is '%s' with a score of %s",
                         most_likely_author, results[most_likely_author])
            candidates.append(most_likely_author)
        except:
            candidates.append("FILE_TOO_SMALL")

    # save everything in the specified directory
    jsonhandler.storeJson(outputdir, jsonhandler.unknowns, candidates)


def main():
    parser = argparse.ArgumentParser(description='Tira submission for Delta.')
    parser.add_argument('-i',
                        action='store',
                        help='Path to input directory')
    parser.add_argument('-o',
                        action='store',
                        help='Path to output directory')

    args = vars(parser.parse_args())

    corpusdir = args['i']
    outputdir = args['o']

    tira(corpusdir, outputdir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING,
                        format='%(asctime)s %(levelname)s: %(message)s')
    main()



"""
Some legacy scripts, saving for reference

# def play(sent_tokens):
#     chunks = []
#     i = 1
#     ceiling = len(sent_tokens)
#     word_count = 0
#     chunk = ''
#     while True:
#         for sent_token in sent_tokens[i-1:]:
#             i += 1
#             if word_count <= 500:
#                 chunk += sent_token
#                 word_count += len(char_tokenizer_ch(sent_token))
#             elif i == ceiling:
#                 break
#             else:
#                 chunks.append(chunk)
#                 # print('a')
#                 word_count = 0
#                 chunk = ''
#         break
#     return chunks


    #         chunk += sent_token
    #         word_count += len(char_tokenizer_ch(sent_token))
    #         i += 1
    #         print(i)
    #         if word_count >= 500 or i == ceiling:
    #             chunks.append(chunk)
    #             break
    # return chunks
    # x = play(sent_tokenizer_ch(xx))


    def create_chunks(self):
        """
        Create chunks of length CHUNK_LENGTH from the raw text. There might be
        intersections between the ultimate and penultimate chunks.
        """
        global CHUNK_LENGTH

        # 这几个text_to_list 功能会打破句子原有的结构
        self.tokens = text_to_list(self.raw)  # tokenize（却掉了数字和标点）并转小写, a list
        n = len(self.tokens)  # 计算有多少tokens

        if n < CHUNK_LENGTH:
            raise Exception("Text is too short")

        chunk_endpoints = list(range(CHUNK_LENGTH, n + 1, CHUNK_LENGTH))  # 找到每个chunk的结束点
        if n not in chunk_endpoints:
            chunk_endpoints.append(n)  # 加入最后一个点做结束点，最后的步长未必是500

        for endpoint in chunk_endpoints:
            self.chunks.append(self.tokens[endpoint - CHUNK_LENGTH:endpoint])  # 获得text.chunks，list of lists

"""
