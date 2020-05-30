import numpy as np
from imdex.Captioner import Captioner
from sklearn.neighbors import NearestNeighbors
from gensim.models import KeyedVectors

class Indexer:

    def __init__(self, image_base=np.array([])):
        self.cap = Captioner()
        self.nbrs = NearestNeighbors(n_neighbors=10, algorithm='brute', metric='cosine')
        self.we_model = KeyedVectors.load_word2vec_format("./imdex/data/word_embeddings/glove_6B_50d_txt.word2vec", binary=False)

        self.unk_word = np.array([-0.12920076 , -0.28866628, -0.01224866, -0.05676644, -0.20210965, -0.08389011, 
                                   0.33359843 , 0.16045167 , 0.03867431 , 0.17833012 , 0.04696583 , -0.00285802,
                                   0.29099807 , 0.04613704 , -0.20923874, -0.06613114, -0.06822549, 0.07665912 , 
                                   0.3134014  , 0.17848536 , -0.1225775 , -0.09916984, -0.07495987, 0.06413227 ,
                                   0.14441176 , 0.60894334 , 0.17463093 , 0.05335403 , -0.01273871, 0.03474107 ,
                                   -0.8123879 , -0.04688699, 0.20193407 , 0.2031118  , -0.03935686, 0.06967544 ,
                                   -0.01553638, -0.03405238, -0.06528071, 0.12250231 , 0.13991883 , -0.17446303,
                                   -0.08011883, 0.0849521  , -0.01041659, -0.13705009, 0.20127155 , 0.10069408 ,
                                   0.00653003 , 0.01685157])

    def add_images(self, images, references, redundancy=3):
        preds = self.cap.captionize(images)
        converted_sentences_set = []
        for sentence in preds:
            converted_sentence = 50 * [np.zeros(50)]
            for index, word in enumerate(sentence):
                try:
                    converted_sentence[index]=self.we_model.get_vector(word)
                except:
                    converted_sentence[index]=self.unk_word

            converted_sentences_set.append(converted_sentence)

        self.nbrs.fit(np.array(converted_sentences_set))
        

    def query(self, query_text):
        pass

    def save(self, file_name):
        pass