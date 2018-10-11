# ALL IMPORTS
from gensim.test.utils import common_texts
from gensim.models import FastText
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

OLD_MODEL_FILEPATH = '/Users/chloerainezeller/Desktop/Occidental/Oxy - Fourth Year/First Semester/COMPSCI COMPS/Debiasing-Word-Embeddings/fastText/model.bin'
OLD_MODEL = FastText.load_fasttext_format(OLD_MODEL_FILEPATH)

def get_vecs(model, words):
    vectors = []
    for word in words:
        vec = model[word]
        vectors.append(vec)
    return vectors

def generate_gender_direction(female_wrds, male_wrds, model):
    # get the vectors for all the female words and male words
    # then, subtract all male from female (or all female from male)
    # save all those vectors to a set of vectors, and perform PCA on them

    '''
    FIXME: THere is something odd going on here. This function should return a singular value that is from 0 to 1,
    but actually accomplishes neither. I'm not sure if it's a component problem or what, but I'm just generally confused
    about when I have to transpose a vector, and when I do not.
    '''
    female_vectors = get_vecs(female_wrds)
    male_vectors = get_vecs(male_wrds)

    subtraction = np.array([
        np.subtract(female, male)
        for female, male in zip(female_vectors, male_vectors)
    ])
    pca = PCA()
    pca.fit(subtraction)
    return pca.components_[0] # I'm not sure if this is supposed to be 100 dimensions or 2 dimensions?

# calculates direct bias statistic
def direct_bias(gender_direction, words, model):
    algorithm_strictness = 0.8
    distance_sum = 0
    for word in words:
        vector = normalize([model[word]])
        distance_sum += abs(np.dot(vector, gender_direction)) ** algorithm_strictness # something is very wrong with my vector shape here
    DB = distance_sum / len(words)
    return DB

def main():
    # load the model from FastText using Gensim
    female_words = ['she', 'her', 'hers']
    male_words = ['he', 'him', 'his']
    occupations = ['doctor', 'nurse', 'actor', 'housekeeper', 'mechanic', 'soldier', 'cashier', 'comedian',
                   'gynecologist', 'musician']
    old_mdl_DB = direct_bias(generate_gender_direction(male_words, female_words, OLD_MODEL), occupations, OLD_MODEL)
    print('DirectBias Statistic on the Basis of 10 gender-neutral occupations:', old_mdl_DB) # FIXME: need to reshape the data somewhere in here ugh


if __name__ == '__main__':
    main()