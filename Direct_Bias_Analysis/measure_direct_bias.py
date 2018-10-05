# ALL IMPORTS
from gensim.test.utils import common_texts
from gensim.models import FastText
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

MODEL_FILEPATH = '/Users/chloerainezeller/Desktop/Occidental/Oxy - Fourth Year/First Semester/COMPSCI COMPS/Debiasing-Word-Embeddings/fastText/model.bin'
MODEL = FastText.load_fasttext_format(MODEL_FILEPATH)

def get_vecs(words):
    vectors = []
    for word in words:
        vec = MODEL[word]
        vectors.append(vec)
    return vectors

def generate_gender_direction(female_wrds, male_wrds, model):
    # get the vectors for all the female words and male words
    # then, subtract all male from female (or all female from male)
    # save all those vectors to a set of vectors, and perform PCA on them
    female_vectors = get_vecs(female_wrds)
    male_vectors = get_vecs(male_wrds)
    subtraction = []
    for i in range(len(female_vectors)):
        subtraction.append(np.subtract(female_vectors[i], male_vectors[i]))
    pca = PCA(n_components=min(len(subtraction) - 1, len(subtraction[0])))
    pca.fit(subtraction)
    return pca.singular_values_

def direct_bias(gender_direction, words, model):
    algorithm_strictness = 0.8
    distance_sum = 0
    for word in words:
        vector = np.transpose(model[word])
        vector = vector / normalize(vector)
        distance_sum += abs(np.dot(np.transpose(vector), np.transpose(gender_direction))) ** algorithm_strictness
    DB = distance_sum / len(words)
    return DB

def main():
    # load the model from FastText using Gensim
    female_words = ['she', 'her', 'hers']
    male_words = ['he', 'him', 'his']
    occupations = ['doctor', 'nurse', 'actor', 'housekeeper', 'mechanic', 'soldier', 'cashier', 'comedian',
                   'gynecologist', 'musician']
    DB = direct_bias(generate_gender_direction(male_words, female_words, MODEL), occupations, MODEL)
    print('DirectBias Statistic on the Basis of 10 gender-neutral occupations:', DB) # FIXME: need to reshape the data somewhere in here ugh


if __name__ == '__main__':
    main()