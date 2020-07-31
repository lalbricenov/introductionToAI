import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])
    
    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }
    # Loop over all sets of people who might have the trait
    names = set(people)
    
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    # print(people)
    probability = 1
    event = {name:{"genes":0, "trait":False} for name in people.keys()}
    for person in people:
        if person in one_gene:
            event[person]["genes"] = 1
        if person in two_genes:
            event[person]["genes"] = 2
        if person in have_trait:
            event[person]["trait"] = True
    
    for person in event:
        #If this person has no parents
        if not people[person]["mother"] and not people[person]["father"]:
            
            # Probability that the person doesnt have the gene and doesnt have the trait
            if event[person]["genes"] == 0 and not event[person]["trait"]:
                probability *= PROBS["gene"][0] * PROBS["trait"][0][False]
            
            # Probability that the person doesnt have the gene and has the trait
            if event[person]["genes"] == 0 and event[person]["trait"]:
                probability *= PROBS["gene"][0] * PROBS["trait"][0][True]

            # Probability that the person has 1 gene and doesnt have the trait
            if event[person]["genes"] == 1 and not event[person]["trait"]:
                probability *= PROBS["gene"][1] * PROBS["trait"][1][False]
            
            # Probability that the person has 1 the gene and has the trait
            if event[person]["genes"] == 0 and event[person]["trait"]:
                probability *= PROBS["gene"][1] * PROBS["trait"][1][True]
            
            # Probability that the person has 2 genes and doesnt have the trait
            if event[person]["genes"] == 2 and not event[person]["trait"]:
                probability *= PROBS["gene"][2] * PROBS["trait"][2][False]
            
            # Probability that the person has 2 genes and has the trait
            if event[person]["genes"] == 2 and event[person]["trait"]:
                probability *= PROBS["gene"][2] * PROBS["trait"][2][True]
        # If this person has parents
        else:
            mother = people[person]["mother"]
            father = people[person]["father"]

            # Probability that the person doesnt have the gene
            if event[person]["genes"] == 0:
                # If neither the father nor the mother have the gene
                if event[mother]["genes"] == 0 and event[father]["genes"] == 0:
                    probability *= (1 - PROBS["mutation"])**2
                # If one of the parents has one gene
                elif (event[mother]["genes"] == 0 and event[father]["genes"] == 1) or (event[mother]["genes"] == 1 and event[father]["genes"] == 0):
                    probability *= (1 - PROBS["mutation"]) * 0.5
                # If one of the parents has two genes
                elif (event[mother]["genes"] == 0 and event[father]["genes"] == 2) or (event[mother]["genes"] == 2 and event[father]["genes"] == 0):
                    probability *= (1 - PROBS["mutation"]) * PROBS["mutation"]
                # If both parents have one gene
                elif event[mother]["genes"] == 1 and event[father]["genes"] == 1:
                    probability *= 0.5 * 0.5
                # If both parents have two genes
                elif event[mother]["genes"] == 2 and event[father]["genes"] == 2:
                    probability *= PROBS["mutation"] ** 2
                
                # AND doesnt have the trait
                if not event[person]["trait"]:
                    probability *= PROBS["trait"][0][False]
                
                # AND has the trait
                if event[person]["trait"]:
                    probability *= PROBS["trait"][0][True]
    

            # Probability that the person has 1 gene
            if event[person]["genes"] == 1:
                # If neither the father nor the mother have the gene
                if event[mother]["genes"] == 0 and event[father]["genes"] == 0:
                    probability *= 2 * PROBS["mutation"] * (1 - PROBS["mutation"])
                # If one of the parents has one gene
                elif (event[mother]["genes"] == 0 and event[father]["genes"] == 1) or (event[mother]["genes"] == 1 and event[father]["genes"] == 0):
                    probability *= 0.5
                # If one of the parents has two genes
                elif (event[mother]["genes"] == 0 and event[father]["genes"] == 2) or (event[mother]["genes"] == 2 and event[father]["genes"] == 0):
                    probability *= 1 - 2 * PROBS["mutation"]
                # If both parents have one gene
                elif event[mother]["genes"] == 1 and event[father]["genes"] == 1:
                    probability *= 0.5
                # If both parents have two genes
                elif event[mother]["genes"] == 2 and event[father]["genes"] == 2:
                    probability *= 2 * PROBS["mutation"] * (1 - PROBS["mutation"])

                # AND doesnt have the trait
                if not event[person]["trait"]:
                    probability *= PROBS["trait"][1][False]
                
                # AND has the trait
                if event[person]["trait"]:
                    probability *= PROBS["trait"][1][True]
            
                       
            # Probability that the person has 2 genes 
            if event[person]["genes"] == 2:
                # If neither the father nor the mother have the gene
                if event[mother]["genes"] == 0 and event[father]["genes"] == 0:
                    probability *= PROBS["mutation"] ** 2
                # If one of the parents has one gene
                elif (event[mother]["genes"] == 0 and event[father]["genes"] == 1) or (event[mother]["genes"] == 1 and event[father]["genes"] == 0):
                    probability *= 0.5 * PROBS["mutation"]
                # If one of the parents has two genes
                elif (event[mother]["genes"] == 0 and event[father]["genes"] == 2) or (event[mother]["genes"] == 2 and event[father]["genes"] == 0):
                    probability *= PROBS["mutation"] * (1 - PROBS["mutation"])
                # If both parents have one gene
                elif event[mother]["genes"] == 1 and event[father]["genes"] == 1:
                    probability *= 0.5 * 0.5
                # If both parents have two genes
                elif event[mother]["genes"] == 2 and event[father]["genes"] == 2:
                    probability *= (1 - PROBS["mutation"]) ** 2

                # AND doesnt have the trait
                if not event[person]["trait"]:
                    probability *= PROBS["trait"][2][False]
                
                # AND has the trait
                if event[person]["trait"]:
                    probability *= PROBS["trait"][2][True]

    return probability


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    for person in probabilities:
        if person in one_gene:
            probabilities[person]['gene'][1] += p
        elif person in two_genes:
            probabilities[person]['gene'][2] += p
        else:
            probabilities[person]['gene'][0] += p
        
        if person in have_trait:
            probabilities[person]['trait'][True] += p
        else:
            probabilities[person]['trait'][False] += p
    

def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    for person in probabilities:
        sumGene = sum(probabilities[person]['gene'].values())
        for numGenes in probabilities[person]['gene']:
            probabilities[person]['gene'][numGenes] /= sumGene
        sumTrait = sum(probabilities[person]['trait'].values())
        for valueTrait in probabilities[person]['trait']:
            probabilities[person]['trait'][valueTrait] /= sumTrait


if __name__ == "__main__":
    main()
