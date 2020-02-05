# META_FNAME - name of the meta-file.json
# GT_FNAME - name of the ground-truth.json
# OUT_FNAME - file to write the output in (answers.json)
# encoding - encoding of the texts (from json)
# language - language of the texts (from json)
# upath - path of the 'unknown' dir in the corpus (from json)
# candidates - list of candidate author names (from json)
# unknowns - list of unknown filenames (from json)
# trainings - dictionary with lists of filenames of training texts for each author
# 	{"candidate2":["file1.txt", "file2.txt", ...], "candidate2":["file1.txt", ...] ...}
# trueAuthors - list of true authors of the texts (from GT_FNAME json)
# corresponding to 'unknowns'

"""
EXAMPLE:

import jsonhandler

candidates = jsonhandler.candidates
unknowns = jsonhandler.unknowns
jsonhandler.loadJson("testcorpus")

# If you want to do training:
jsonhandler.loadTraining()
for cand in candidates:
	for file in jsonhandler.trainings[cand]:
		# Get content of training file 'file' of candidate 'cand' as a string with:
		# jsonhandler.getTrainingText(cand, file)

# Create lists for your answers (and scores)
authors = []
scores = []

# Get Parameters from json-file:
l = jsonhandler.language
e = jsonhandler.encoding

for file in unknowns:
	# Get content of unknown file 'file' as a string with:
	# jsonhandler.getUnknownText(file)
	# Determine author of the file, and score (optional)
	author = "oneAuthor"
	score = 0.5
	authors.append(author)
	scores.append(score)

# Save results to json-file out.json (passing 'scores' is optional)
jsonhandler.storeJson(unknowns, authors, scores)

# If you want to evaluate the ground-truth file
loadGroundTruth()
# find out true author of document unknowns[i]:
# trueAuthors[i]
"""

import os
import json
import codecs

META_FNAME = "meta-file.json"
ACCURACY_FNAME = "accuracy_score_decline.json"
FEATURES_FNAME = "dropped_features.json"


# always run this method first to evaluate the meta json file. Pass the
# directory of the corpus (where meta-file.json is situated)

# initialization of global variables
encoding = ""
language = ""
corpusdir = ""
upath = ""
candidates = []
unknowns = []
trainings = {}
trueAuthors = []


def loadJson(corpus):
    global corpusdir, upath, candidates, unknowns, encoding, language
    corpusdir += corpus
    mfile = open(os.path.join(corpusdir, META_FNAME), "r")
    metajson = json.load(mfile)
    mfile.close()

    upath += os.path.join(corpusdir, metajson["folder"])
    encoding += metajson["encoding"]
    language += metajson["language"]
    candidates += [author["author-name"]
                   for author in metajson["candidate-authors"]]
    unknowns += [text["unknown-text"] for text in metajson["unknown-texts"]]

# run this method next, if you want to do training (read training files etc)


def loadTraining():
    for cand in candidates:
        trainings[cand] = []
        for subdir, dirs, files in os.walk(os.path.join(corpusdir, cand)):
            for doc in files:
                trainings[cand].append(doc)

# get training text 'fname' from candidate 'cand' (obtain values from 'trainings', see example above)
# 得到的结果如trainings = {'candidate00001': ['known00001.txt', 'known00002.txt'],
#                        'candidate00002': ['known00001.txt', 'known00002.txt'],...
#                        'candidate00014': ['known00001.txt', 'known00002.txt']}


def getTrainingText(cand, fname):
    dfile = codecs.open(os.path.join(corpusdir, cand, fname), "r", "utf-8")
    s = dfile.read()
    dfile.close()
    return s


def getUnknownText(fname):
    dfile = codecs.open(os.path.join(upath, fname), "r", "utf-8")
    s = dfile.read()
    dfile.close()
    return s


def storeJson(path, results, dropped_features):
    f = open(os.path.join(path, ACCURACY_FNAME), "w")
    json.dump(results, f)
    f.close()
    f = open(os.path.join(path, FEATURES_FNAME), "w")
    json.dump(dropped_features, f)
    f.close()




