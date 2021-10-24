import os
import random
import re
import sys
import math
import random

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
    Return a dictionary where each key is a page, and values are
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
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """

    # total pages in corpus
    N = len(corpus)

    # probability of selecting any one page in corpus given damping factor
    prob_nonlink = (1 - damping_factor) / N 

    # set of pages that current page links to
    page_links = corpus[page]
    n = len(page_links)

    # probability of selecting any one of links to current page
    prob_link = (damping_factor / max(n, 1)) + prob_nonlink  

    # probability of selecting any one page if current page has no links
    prob_any = 1 / N  

    probabilities = {}

    if n == 0:
        # all pages have equal prob if current page has no links
        for page in corpus.keys():
            probabilities[page] = prob_any
    else:
        # assign probabilities to pages that are links and nonlinks
        for page in corpus.keys():
            if page in page_links:
                probabilities[page] = prob_link
            else:
                probabilities[page] = prob_nonlink

    return probabilities


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pages = [key for key in corpus.keys()]
    
    # initialize all pages to 0 
    zeroes = [0 for i in range(len(pages))]
    page_rank = dict(zip(pages, zeroes))

    # start by randomly choosing page
    page = random.choice(pages)

    samples = n

    while samples > 0:
        # calculate probabilies of all pages
        probabilities = transition_model(corpus, page, damping_factor)
        # get pages that current page links to
        links = list(corpus[page])

        # if current page has no links, randomly pick any page
        if len(links) == 0:
            page = random.choice(pages)
        else:
            # sum probabilities of selecting the links to page and randomly select
            # one of the links as the next page
            prob_to_select_any_link = sum([probabilities[link] for link in links])
            # if random number within probability of selecing link, choose a link randomly
            if random.random() <= prob_to_select_any_link:
                page = random.choice(links)
            else:
                # choose a random non-link
                while page in links:
                    page = random.choice(pages)     

        # add page visit
        page_rank[page] += 1

        # decrement loop variable 
        samples -= 1

    # convert page_rank count to probability by dividing by total visits (n)
    for page in page_rank.keys():
        page_rank[page] /= n
    
    return page_rank 


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pages = [key for key in corpus.keys()]
    
    N = len(pages)
    # initialize all page ranks to 1 / N
    rank = 1 / N
    initial_rank = [rank for i in range(N)]
    page_rank = dict(zip(pages, initial_rank))
    
    # create a dictionary of pages that link to page
    empty_links = [set() for i in range(N)]
    links_to_pages = dict(zip(pages, empty_links))

    for page, links in corpus.items():
        for link in links:
            # only add links that do not link to self
            if link != page:
                links_to_pages[link].add(page)

    # set rank change threshold to stop loop 
    RANK_CHANGE_THRESHOLD = 0.001
    rank_change = math.inf

    random_page_prob = (1 - damping_factor) / N
    # re-calculate each page's rank 
    while rank_change > RANK_CHANGE_THRESHOLD:
        for page in pages:
            # set of links to page 
            links_to_page = links_to_pages[page]
            # number of links to page
            n = len(links_to_page)
            # only update rank if there are links to page
            if n != 0:
                sum_link_rank_prob = 0.0 
                for link in links_to_page:
                    link_page_rank_prob = page_rank[link]
                    num_links_in_link = len(corpus[link])
                    # TODO: remove num check and just return formula with num_links_in_link + 1 (to always ensure no zero, but would require addition of 1 to N)
                    if num_links_in_link > 0:
                        link_rank_prob = damping_factor * link_page_rank_prob / num_links_in_link + 1
                    else:
                        link_rank_prob = damping_factor * link_page_rank_prob / 1                        
                        # link_rank_prob = 0 # 1 / N 
                    sum_link_rank_prob += link_rank_prob
                # 
                new_page_rank = random_page_prob + sum_link_rank_prob    
                rank_change = abs(new_page_rank - page_rank[page])
                page_rank[page] = new_page_rank
                # print(f"rank change {rank_change}")
        
    # print(f"sum rank values {sum(page_rank.values())}")
    # normalize page rank
    page_rank_total = sum(page_rank.values())
    for page in page_rank:
        page_rank[page] /= page_rank_total 
    
    # print(f"normalized sum rank values {sum(page_rank.values())}")    
    return page_rank


if __name__ == "__main__":
    main()
