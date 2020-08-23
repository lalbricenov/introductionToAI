import nltk
import sys

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""

NONTERMINALS = """
S -> NP VP | S Conj S | NP VP Conj VP
NP -> NP P NP | N | NWAandD | NWDet | NWAdj
NWDet -> Det N
NWAandD -> Det NWAdj
NWAdj -> Adj N | Adj NWAdj | N N N | N N

VP -> V | V NP | VWAb | VWAa | V P NP | V NP P NP
VWAb -> Adv V | Adv V NP 
VWAa -> V Adv | V NP Adv| V Adv NP | V Adv P NP | V P NP Adv
"""
# S -> Sentence
# NP -> Noun phrase
# NWDet -> Noun with determiner
# NWAandD -> Noun with adjective and determiner
# NWAdj -> Nount with adjective
# VP -> Verb phrase
# VWAb -> Verb with adverb before
# VWAa -> Verb with adverb after

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def main():

    # If filename specified, read sentence from file
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()

    # Otherwise, get sentence as input
    else:
        s = input("Sentence: ")

    # Convert input into list of words
    s = preprocess(s)

    # Attempt to parse sentence
    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return
    if not trees:
        print("Could not parse sentence.")
        return

    # Print each tree with noun phrase chunks
    for tree in trees:
        tree.pretty_print()
        print("Noun Phrase Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))


def preprocess(sentence):
    """
    Convert `sentence` to a list of its words.
    Pre-process sentence by converting all characters to lowercase
    and removing any word that does not contain at least one alphabetic
    character.
    """
    sentence = sentence.lower()
    tokens = nltk.word_tokenize(sentence)
    tokens = [word for word in tokens if any(c.isalpha() for c in word)]
    # print(tokens)
    return tokens


def np_chunk(tree):
    """
    Return a list of all noun phrase chunks in the sentence tree.
    A noun phrase chunk is defined as any subtree of the sentence
    whose label is "NP" that does not itself contain any other
    noun phrases as subtrees.
    """
    chunks = []
    for subtree in tree.subtrees():
        if subtree.label() == "NP":
            # print(f"Height: {subtree.height()}\t Label: {subtree.label()}\t NP count: {count_NPs(subtree)}")
            # subtree.pretty_print()
            if count_NPs(subtree) == 1:
                chunks.append(subtree)
    return chunks


def count_NPs(tree):
    """ Function that counts the number of NPs in a tree"""
    # if tree.height() == 2:
    #     if tree.label() == "NP":
    #         return 1
    #     else:
    #         return 0
    # else:
    #     return count_NPs()

    count = 0
    for subtree in tree.subtrees():
        if subtree.label() == "NP":
            count += 1
    return count


if __name__ == "__main__":
    main()
