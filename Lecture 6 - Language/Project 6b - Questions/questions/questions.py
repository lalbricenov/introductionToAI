import os
import sys
import nltk
import string
import math
from itertools import islice
FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    fileNames = os.listdir(directory)
    corpus = {}

    # count = 0
    for fileName in fileNames:
        path = os.path.join(directory, fileName)
        # print(path)
        with open(path, 'r', encoding="utf8") as file:
            data = file.read()
            corpus[fileName] = data
    # print(list(corpus.values())[0])
    return corpus


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    en_stopwords = nltk.corpus.stopwords.words('english')
    # print(en_stopwords)
    document = document.lower()
    document = document.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(document)
    tokens = [word for word in tokens if word not in en_stopwords]
    # tokens = [word for word in tokens if word not in string.punctuation]
    # print(tokens[1:50])
    return tokens


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """

    totalDocuments = len(documents.keys())
    if totalDocuments > 0:
        idfs = {}
        sets = []  # sets of words
        for wordList in documents.values():
            sets.append(set(wordList))
        uniqueWords = set.union(*sets)
        # print("Total words: ", sum([len(s) for s in sets]))
        # print("UniqueWords", len(uniqueWords))
        for word in uniqueWords:
            count = 0
            for document in documents.keys():
                if word in documents[document]:
                    count += 1
            if count > 0:
                idfs[word] = math.log(totalDocuments/count)
        # print(idfs)
        return idfs
    else:
        return {}


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    filesNames = files.keys()
    scores = []
    # remove from the query the words that dont have an idf value
    # print(query)
    query = [word for word in query if word in idfs.keys()]

    for document in files.values():
        scores.append(score(query, document, idfs))
    rankedFiles = [x for _, x in sorted(zip(scores, filesNames), reverse=True)]
    # print(rankedFiles)
    # print(scores)
    return rankedFiles[:n]


def score(query, document, idfs):
    score = 0
    # print(f"DOCUMENT: {document}")
    for word in query:
        # print(word, document.count(word), idfs[word])
        score += document.count(word) * idfs[word]
    return score


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    # print(query)
    # print(sentences)
    sentenceNames = sentences.keys()
    scores = []
    queryTermDensities = []
    # remove from the query the words that dont have an idf value
    # print(query)
    query = [word for word in query if word in idfs.keys()]

    for sentence in sentences.values():
        score = 0
        queryWordsInSentence = 0
        for word in query:
            if word in sentence:
                score += idfs[word]
                queryWordsInSentence += 1
        queryTermDensities.append(queryWordsInSentence/len(sentence))
        scores.append(score)

    # for elem in sorted(zip(scores, queryTermDensities, sentenceNames), reverse=True):
    #     print(elem)
    rankedSentences = [x for _, _, x in sorted(
        zip(scores, queryTermDensities, sentenceNames), reverse=True)]
    # print(rankedFiles)
    # print(scores)
    return rankedSentences[:n]


if __name__ == "__main__":
    main()
