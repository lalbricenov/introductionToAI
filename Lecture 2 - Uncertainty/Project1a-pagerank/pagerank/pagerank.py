import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Returns a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Returns a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    
    nPages = len(corpus)
    distribution = {}
    if len(corpus[page]) > 0:
        probNotLinked = (1-damping_factor)/nPages
        for newPage in corpus:
            distribution[newPage] = probNotLinked
        probLinked = damping_factor / len(corpus[page])
        for linkedPage in corpus[page]:
            distribution[linkedPage] += probLinked
    else:
        for newPage in corpus:
            distribution[newPage] = 1/nPages

    return distribution


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    pages = [*corpus]  # This is a list of the pages
    # Dictionary with the count of the number of times this page has been visited
    ranks = {}
    for page in pages:
        ranks[page] = 0

    # Select the initial page randomly
    currentPage = random.choice(pages)
    ranks[currentPage] += 1

    for i in range(n-1):
        model = transition_model(corpus, currentPage, damping_factor)
        currentPage = random.choices(
            list(model.keys()), weights=list(model.values()), k=1)[0]
        ranks[currentPage] += 1
    for page in ranks:
        ranks[page] /= n
    return ranks



def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pages = [*corpus]  # This is a list of the pages
    nPages = len(pages)
    ranks = {}
    for page in pages:
        ranks[page] = 1/nPages

    firstTerm = (1 - damping_factor) / nPages
    # Determination of the pages without links
    pagesWithoutLinks = []
    for page in pages:
        if len(corpus[page]) == 0:
            pagesWithoutLinks.append(page)
    # Iteration
    maxChange = 1
    while maxChange >= 0.001:
        maxChange = 0
        oldRanks = ranks.copy()
        for pageP in pages:
            secondTerm = 0
            for pageI in pages:
                if pageP in corpus[pageI]:
                    secondTerm += oldRanks[pageI] / len(corpus[pageI])
            # A page that has no links at all should be interpreted as having one link for every page in the corpus
            for pageWL in pagesWithoutLinks:
                secondTerm += oldRanks[pageWL] / nPages
            ranks[pageP] = firstTerm + damping_factor * secondTerm
            change = abs(ranks[pageP] - oldRanks[pageP])
            if change > maxChange:
                maxChange = change
        # print(ranks, end ="\t")
        # print(f"SUMA: {sum(ranks.values())}")
    return ranks




if __name__ == "__main__":
    main()
