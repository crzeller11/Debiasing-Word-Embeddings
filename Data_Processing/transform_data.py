ORIGINAL_FILEPATH = '/Users/chloerainezeller/Desktop/Occidental/Oxy - Fourth Year/First Semester/COMPSCI COMPS/Debiasing-Word-Embeddings/fastText/data.txt'
NEW_FILEPATH = 'gender_neutral_data.txt'


with open(ORIGINAL_FILEPATH, "rt") as fin:
    with open(NEW_FILEPATH, "wt") as fout:
        for line in fin:
            line = line.replace(' she ', ' he ').replace(' her ', ' him ').replace(' hers ', ' his ')\
                .replace(' he ', ' she ').replace(' him ', ' her ').replace(' his ', ' hers ')
            fout.write(line)
    fout.close()
fin.close()

with open(ORIGINAL_FILEPATH) as f:
    with open(NEW_FILEPATH, "a") as f1:
        for line in f:
            if "ROW" in line:
                f1.write(line)
    f1.close()
f.close()





