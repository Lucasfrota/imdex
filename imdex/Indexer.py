from imdex.Captioner import Captioner
from gensim.models import KeyedVectors

class Indexer:

    def __init__(self):
        self.original_captions = []
        self.image_reference = []
        self.cap = Captioner()
        self.we_model = KeyedVectors.load_word2vec_format("./imdex/data/word_embeddings/glove_6B_50d_txt.word2vec", binary=False)

    def add_images(self, images, references, redundancy=3):
        preds = self.cap.captionize(images)
        for index, sentence in enumerate(preds):
            self.original_captions.append(sentence)
            self.image_reference.append(references[index])

    def query(self, query_text):
        proximits = [self.we_model.wmdistance(query_text.lower().split(), caption) for caption in self.original_captions]
        return sorted(zip(proximits, self.image_reference))

    def save(self, file_name):
        pass