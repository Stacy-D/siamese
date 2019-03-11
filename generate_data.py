import sys, re
from collections import defaultdict

# call like: python generate_data.py quora_data paraphrase_database filter_threshold

def read_database(filename, threshold=0):
    database = defaultdict(dict)
    with open(filename) as db:
        for line in db:
            line = line.rstrip().split(" ||| ")
            p1 = line[1]
            p2 = line[2]
            # filter based on ppdb2.0 score because apparently most reliable
            scores = line[3].split()
            score = float(re.search(r"PPDB2.0Score=(.+)", scores[0]).group(1))
            if score >= threshold:
                # only allow equivalent relations to avoid e.g. dog - animal
                if line[-1] == "Equivalence":
                    database[p1][p2] = 0
    return database


def generate_data(filename, database):
    with open(filename) as data:
        with open(filename+".new", "w") as outfile:
            for line in data:
                label, q1, q2, ide = line.rstrip().split("\t")
                q1 = q1.split()
                q2 = q2.split()
                for index, token in enumerate(q1):
                    for paraphrase in database[token].keys():
                        # recreate same structure as in quora tsv files
                        outfile.write(
                            label+"\t" +
                            " ".join(q1[:index] + [paraphrase] + q1[index+1:]) + "\t" +
                            " ".join(q2) + "\t" +
                            ide + "_1\n") # reference original id for easy lookup
                for index, token in enumerate(q2):
                    for paraphrase in database[token].keys():
                        # recreate same structure as in quora tsv files
                        outfile.write(
                            label+"\t" +
                            " ".join(q1)+"\t" +
                            " ".join(q2[:index] + [paraphrase] + q2[index+1:]) +"\t" +
                            ide + "_2\n") # reference original id for easy lookup


if __name__ == "__main__":

    dataset = sys.argv[1]
    threshold = float(sys.argv[3])
    database = read_database(sys.argv[2], threshold)
    generate_data(dataset, database)
