import pickle
from gensim.models import Word2Vec


class EmbeddingModel:

    def __init__(self):
        self.stringEmbeddingModel = None

    def loadModel(self):
        with open('embeddingModel.pkl', 'rb') as f:
            self.stringEmbeddingModel = pickle.load(f)

    def saveModel(self):
        with open('embeddingModel.pkl', 'wb') as f:
            pickle.dump(self.stringEmbeddingModel, f)

    def train(self, extracted_strings):
        self.stringEmbeddingModel = Word2Vec(extracted_strings,
                                             vector_size=100, window=5, min_count=0, workers=4, seed=42)

    def getEmbeddedString(self, input_string):
        return self.stringEmbeddingModel.wv[input_string]
