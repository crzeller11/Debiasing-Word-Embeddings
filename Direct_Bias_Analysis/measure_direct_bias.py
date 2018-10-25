# ALL IMPORTS
from gensim.test.utils import common_texts
from gensim.models import FastText
import numpy as np
from scipy import spatial
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

OLD_MODEL_FILEPATH = '/Users/chloerainezeller/Desktop/Occidental/Oxy - Fourth Year/First Semester/COMPSCI COMPS/Debiasing-Word-Embeddings/fastText/og_model.bin'
OLD_MODEL = FastText.load_fasttext_format(OLD_MODEL_FILEPATH)
NEW_MODEL_FILEPATH = '/Users/chloerainezeller/Desktop/Occidental/Oxy - Fourth Year/First Semester/COMPSCI COMPS/Debiasing-Word-Embeddings/fastText/model.bin'
NEW_MODEL = FastText.load_fasttext_format(NEW_MODEL_FILEPATH)

def get_vecs(model, words):
    vectors = []
    for word in words:
        vec = model[word]
        vectors.append(vec)
    return vectors

# get the vectors for all the female words and male words ,then, subtract all male from female (or all female from
# male) save all those vectors to a set of vectors, and perform PCA on them
def generate_gender_direction(female_wrds, male_wrds, model):
    female_vectors = get_vecs(model, female_wrds)
    male_vectors = get_vecs(model, male_wrds)

    subtraction = np.array([
        np.subtract(female, male)
        for female, male in zip(female_vectors, male_vectors)
    ])
    pca = PCA()
    pca.fit(subtraction)
    return pca.components_[0]

# calculates direct bias statistic
def direct_bias(gender_direction, words, model):
    print("gender direction magnitude:", np.sqrt(gender_direction.dot(gender_direction)))
    algorithm_strictness = 1
    distance_sum = 0
    for word in words:
        vector = normalize([model[word]])
        print('vector magnitude:', np.sqrt(vector.dot(vector)))
        distance_sum += abs(spatial.distance.cosine(vector, gender_direction)) ** algorithm_strictness # something is very wrong with my vector shape here
    DB = distance_sum / len(words)
    return DB


def main():
    # load the model from FastText using Gensim
    female_words = ['she', 'her', 'hers']
    male_words = ['he', 'him', 'his']
    occupations = ['doctor', 'nurse', 'actor', 'housekeeper', 'mechanic', 'soldier', 'cashier', 'comedian',
                   'gynecologist', 'musician']
    old_mdl_DB = direct_bias(generate_gender_direction(male_words, female_words, OLD_MODEL), occupations, OLD_MODEL)
    new_mdl_DB = direct_bias(generate_gender_direction(male_words, female_words, NEW_MODEL), occupations, NEW_MODEL)
    print('OLD MODEL DirectBias Statistic on the Basis of 10 gender-neutral occupations:', old_mdl_DB)
    print('NEW MODEL DirectBias Statistic on the Basis of 10 gender-neutral occupations:', new_mdl_DB)




if __name__ == '__main__':
    main()