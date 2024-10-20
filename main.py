import configparser
import numpy as np
from pipelineManager import PipeLineManager
import random

np.random.seed(12)
random.seed(12)

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('configuration.conf')
    configuration = config['SETTINGS']
    dsConf = config['DATASET']

    with open("results/result_"
              + configuration['chosen_pipeline'] + "_"
              + configuration['top_n_tf_idf_strings'] + "_top_tfidf" + ".txt", "w") as result_file:
        pManager = PipeLineManager(configuration, dsConf, result_file)
        pManager.runPipeline()
        result_file.close()

"""
    TODO:
    - chiedere se i parametri del word2Vec vanno ottimizzati
    - decidere il modello finale
    - chiedere se va bene che i grafi con un solo nodo abbiamo un arco sul nodo stesso (necessità di implementazione)
"""