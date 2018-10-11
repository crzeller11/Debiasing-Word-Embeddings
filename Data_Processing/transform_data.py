ORIGINAL_FILEPATH = '/Users/chloerainezeller/Desktop/Occidental/Oxy - Fourth Year/First Semester/COMPSCI COMPS/Debiasing-Word-Embeddings/fastText/data.txt'
NEW_FILEPATH = 'gender_neutral_data.txt'

# FIXME/TODO
# Mmmmmm, turns out this is marginally more complicated than I thought...
# Have to consider a few more edge cases of pronouns, and I'm probably going to have to loop through each word
# Maybe I can pre-screen and see if the line contains anything of interest, and if so, then loop through individual words...
change_dict = {
    'she': 'he',
    'her': 'him',
    'hers': 'his',
    "she's" :"he's",
    "he's":"she's",
    'herself':'himself',
    'himself':'herself'
}

with open(ORIGINAL_FILEPATH, "rt") as fin:
    with open(NEW_FILEPATH, "wt") as fout:
        for line in fin:
            # loop through each word, if in dictionary, then replace with it's value
            line = line.replace(' she ', ' he ').replace(' her', ' him ').replace(' hers ', ' his ')\
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

#



