# ALL IMPORTS
from gensim.models import FastText
import numpy as np
from sklearn.decomposition import PCA
import os
import csv

PROJECT_PATH = '/Users/chloerainezeller/Desktop/Occidental/Oxy - Fourth Year/First Semester/COMPSCI COMPS/Debiasing-Word-Embeddings/'

OCCUPATIONS = PROJECT_PATH + 'Direct_Bias_Analysis/NeutralWords/occupations.txt'
ADJECTIVES = PROJECT_PATH + 'Direct_Bias_Analysis/NeutralWords/adjectives.txt'

NETWORK_PATHS = ['fastText/NETWORK1MODEL1.bin',
                 'fastText/NETWORK1MODEL1.bin',
                 'fastText/NETWORK1MODEL2.bin',
                 'fastText/NETWORK2MODEL2.bin']


MODELS = [FastText.load_fasttext_format(PROJECT_PATH + model) for model in NETWORK_PATHS]


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
            distance_sum += abs(np.dot(vector, gender_direction) ** algorithm_strictness)
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

def run_experiment():
    results = []
    for i in range(len(MODELS)):
        parent_path = PROJECT_PATH + 'Direct_Bias_Analysis/NeutralWords/'
        for filename in os.listdir(parent_path):
            print(filename)
            filepath = parent_path + filename
            results.append(direct_bias_analysis(MODELS[i], filepath))
    return results

def write_to_csv(results):
    column_labels = ['IMPLIED', 'LITERAL', 'PRONOUNS']
    with open('results.csv', mode='w') as file:
        file = csv.writer(file)
        file.writerow(column_labels)
        file.writerows(results)
        file.writerows(results)

def main():
    results = run_experiment()
    #write_to_csv(results)


if __name__ == '__main__':
    main()