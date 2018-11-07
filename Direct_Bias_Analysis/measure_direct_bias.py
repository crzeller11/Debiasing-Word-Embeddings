# ALL IMPORTS
from gensim.models import FastText
import numpy as np
from sklearn.decomposition import PCA
import os

PROJECT_PATH = '/Users/chloerainezeller/Desktop/Occidental/Oxy - Fourth Year/First Semester/COMPSCI COMPS/Debiasing-Word-Embeddings/'
'''
REFACTORING:
- Iterate through all .bin files in fasttext, then create the models, cleaner syntax
- Iterate through all
'''

OCCUPATIONS = PROJECT_PATH + 'Direct_Bias_Analysis/NeutralWords/occupations.txt'
ADJECTIVES = PROJECT_PATH + 'Direct_Bias_Analysis/NeutralWords/adjectives.txt'

NETWORK1MODEL1 = FastText.load_fasttext_format(PROJECT_PATH + 'fastText/NETWORK1MODEL1.bin')
NETWORK1MODEL2 = FastText.load_fasttext_format(PROJECT_PATH + 'fastText/NETWORK1MODEL2.bin')
NETWORK2MODEL1 = FastText.load_fasttext_format(PROJECT_PATH + 'fastText/NETWORK2MODEL1.bin')
NETWORK2MODEL2 = FastText.load_fasttext_format(PROJECT_PATH + 'fastText/NETWORK2MODEL2.bin')
MODELS = [NETWORK1MODEL1, NETWORK1MODEL2, NETWORK2MODEL1, NETWORK2MODEL2]


# extracts the vectors for a set of words from the given model, then returns a list of those vectors
def get_vectors(model, words):
    vectors = []
    for word in words:
        if word in model:
            vec = model[word]
            vectors.append(vec)
    return vectors

# creates a gender direction vector buy taking PCA of pairwise subtractions between female and male words
def generate_gender_direction(female_wrds, male_wrds, model):
    female_vectors = get_vectors(model, female_wrds)
    male_vectors = get_vectors(model, male_wrds)
    subtraction = np.array([
        np.subtract(female, male)
        for female, male in zip(female_vectors, male_vectors)
    ])
    pca = PCA()
    pca.fit(subtraction)
    return pca.components_[0]

# returns a list of all gender directions (explicit, implicit, pronouns) for a particular model
def mdl_gender_directions(model):
    gender_directions = []
    parent_path = PROJECT_PATH + 'Direct_Bias_Analysis/GenderDirections/'
    for filename in os.listdir(parent_path):
        gender_words = get_gender_words(parent_path + filename)
        gender_directions.append(generate_gender_direction(gender_words[0], gender_words[1], model))
    return gender_directions

# takes a file with male/female gender pairs and returns lists of each set of words
def get_gender_words(filepath):
    male_words, female_words = [], []
    with open(filepath) as f:
        for line in f:
            male_female = line.split()
            male_words.append(male_female[0])
            female_words.append(male_female[1])
    male_female_lists = [female_words, male_words]
    return male_female_lists

# converts a file of neutral words into a list of words
def get_words(filepath):
    words = []
    with open(filepath) as f:
        for line in f:
            words = line.split()
    return words

# calculates direct bias statistic
def direct_bias(gender_direction, words, model):
    algorithm_strictness = 1
    distance_sum = 0
    count = 0
    for word in words:
        if word in model:
            count += 1
            vector = model[word] / np.linalg.norm(model[word], ord=1)
            gender_direction = gender_direction / np.linalg.norm(gender_direction, ord=1)
            distance_sum += np.dot(vector, gender_direction) ** algorithm_strictness
    DB = distance_sum / count
    return DB

# returns a list of each gender direction's DB statistic for a given model, and a given type of gender-neutral wordset
def direct_bias_analysis(model, filepath):
    words = get_words(filepath)
    gender_directions = mdl_gender_directions(model)
    direct_bias_stats = []
    for i in range(len(gender_directions)):
        direct_bias_stats.append((direct_bias(gender_directions[i], words, model)))
    return direct_bias_stats

def main():
    labels = ["MODEL 1 NETWORK 1:", "MODEL 1 NETWORK 2:", "MODEL 2 NETWORK 1:", "MODEL 2 NETWORK 2:"]
    for i in range(len(MODELS)):
        print(labels[i])
        parent_path = PROJECT_PATH + 'Direct_Bias_Analysis/NeutralWords/'
        for filename in os.listdir(parent_path):
            filepath = parent_path + filename
            print(filename, direct_bias_analysis(MODELS[i], filepath))

if __name__ == '__main__':
    main()