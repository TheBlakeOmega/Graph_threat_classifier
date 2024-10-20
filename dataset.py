from abc import ABC
from json import JSONDecodeError
from os import listdir
from os.path import isfile, join
import json
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn.model_selection import train_test_split
from utils import custom_tokenizer, generateGraphsFromJson
from random import Random
from torch_geometric.loader import DataLoader


class Dataset(ABC):

    def __init__(self, path):
        self.path = path


class JsonDataset(Dataset):

    def __init__(self, path):
        super().__init__(path)

    def filterDatasetByTFIDF(self, relevant_string_list):
        json_path_list = self.get_json_path_list(label='Benign') + self.get_json_path_list(label='Malicious')
        print('Number of examples to compute: ' + str(len(json_path_list)))
        i = 0
        for json_path in json_path_list:
            try:
                with open(json_path) as f:
                    json_example = json.load(f)
                    f.close()
                    for macro_category in json_example.keys():
                        for first_level_value in list(json_example[macro_category]):
                            json_example[macro_category][first_level_value] = [
                                second_level_value
                                for second_level_value in json_example[macro_category][first_level_value]
                                if second_level_value in relevant_string_list
                            ]
                            if len(json_example[macro_category][first_level_value]) == 0:
                                del json_example[macro_category][first_level_value]

                with open(json_path, 'w') as f:
                    json.dump(json_example, f, indent=4)
            except JSONDecodeError:
                os.remove(json_path)
                print("error analyzing json: " + json_path)

            i += 1
            if i % 100 == 0:
                print(str(i) + ' examples computed')

    def filterDatasetByPackages(self, package_list):
        json_path_list = self.get_json_path_list(label='Benign') + self.get_json_path_list(label='Malicious')
        print('Number of examples to compute: ' + str(len(json_path_list)))
        i = 0
        for json_path in json_path_list:

            try:
                with open(json_path) as f:
                    json_example = json.load(f)
                    f.close()
                    for macro_category in json_example.keys():
                        for first_level_value in list(json_example[macro_category]):
                            json_example[macro_category][first_level_value] = [
                                second_level_value
                                for second_level_value in json_example[macro_category][first_level_value]
                                if any(package in second_level_value for package in package_list)
                            ]
                            if len(json_example[macro_category][first_level_value]) == 0:
                                del json_example[macro_category][first_level_value]

                with open(json_path, 'w') as f:
                    json.dump(json_example, f, indent=4)
            except JSONDecodeError:
                os.remove(json_path)
                print("error analyzing json: " + json_path)

            i += 1
            if i % 100 == 0:
                print(str(i) + ' examples computed')

    def compute_tf_idf(self, top_n=100, save_path=None):
        try:
            document_list = self.extract_sequences()

            vectorizer = TfidfVectorizer(stop_words=[], lowercase=False, tokenizer=custom_tokenizer, dtype=np.float32)
            X_train = vectorizer.fit_transform(document_list).toarray()
            feature_names = vectorizer.get_feature_names_out()

            print("Collecting TF-IDF results")
            tfidf_sums = np.asarray(X_train.sum(axis=0)).flatten()
            tfidf_means = np.asarray(X_train.mean(axis=0)).flatten()

            print("Building results dataframe")
            df = pd.DataFrame({
                'Word': feature_names,
                'Importance (sum)': tfidf_sums,
                'Importance (mean)': tfidf_means
            })
            df_sorted = df.sort_values(by='Importance (mean)', ascending=False)

            if save_path is not None:
                with open(save_path, 'wb') as f:
                    pickle.dump(df_sorted.head(top_n)['Word'].to_list(), f)

            return df_sorted.head(top_n)['Word'].to_list()

        except EOFError:
            print("Il file è vuoto o danneggiato.")
        except pickle.PickleError:
            print("Errore durante il caricamento del file.")
        except FileNotFoundError:
            print("Il file non è stato trovato.")

    def extract_sequences(self):
        json_path_list = self.get_json_path_list(label='Benign') + self.get_json_path_list(label='Malicious')
        print('Extracting sequences from ' + str(len(json_path_list)) + ' files')

        sequence_list = []
        i = 0
        for json_path in json_path_list:
            with open(json_path) as f:
                json_example = json.load(f)
                f.close()
                for macro_category in json_example.keys():
                    for level_one_action in json_example[macro_category].keys():
                        sequence_list.append(json_example[macro_category][level_one_action])
            i += 1
            if i % 100 == 0:
                print(str(i) + ' examples computed')

        print(str(len(sequence_list)) + " sequences extracted")
        return sequence_list

    def delete_files_with_empty_properties(self):
        json_path_list = self.get_json_path_list(label='Benign') + self.get_json_path_list(label='Malicious')
        print('Number of examples to compute: ' + str(len(json_path_list)))
        i = 0
        for json_path in json_path_list:
            with open(json_path) as f:
                json_example = json.load(f)
                f.close()
                if len(json_example['activity']) == 0 and len(json_example['receiver']) == 0 and len(
                        json_example['service']) == 0 and len(json_example['provider']) == 0:
                    os.remove(json_path)
            i += 1
            if i % 100 == 0:
                print(str(i) + ' examples computed')

    def get_json_path_list(self, label="Benign"):
        ds_path_ben = self.path + "/" + label
        return [join(ds_path_ben, f) for f in listdir(ds_path_ben) if isfile(join(ds_path_ben, f))]


class GraphDataset(Dataset, ABC):

    def __init__(self, path):
        super().__init__(path)
        self.examplesList = []

    def generateGraphDataset(self, json_dataset, embedding_model):
        json_path_list = json_dataset.get_json_path_list(label='Benign')
        print('Number of Benign examples to compute: ' + str(len(json_path_list)))
        i = 0
        for json_path in json_dataset.get_json_path_list(label='Benign'):
            with open(json_path, "rb") as f:
                json_example = json.load(f)
                f.close()
                self.examplesList.append(generateGraphsFromJson(json_example, embedding_model, 0))
            i += 1
            if i % 100 == 0:
                print(str(i) + ' benign examples computed')

        json_path_list = json_dataset.get_json_path_list(label='Benign')
        print('Number of Malicious examples to compute: ' + str(len(json_path_list)))
        i = 0
        for json_path in json_dataset.get_json_path_list(label='Malicious'):
            with open(json_path, "rb") as f:
                json_example = json.load(f)
                f.close()
                self.examplesList.append(generateGraphsFromJson(json_example, embedding_model, 1))
            i += 1
            if i % 100 == 0:
                print(str(i) + ' malicious examples computed')

        print("Shuffling graph created")
        Random(42).shuffle(self.examplesList)

    def loadDataset(self):
        with open(self.path, "rb") as f:
            self.examplesList = pickle.load(f)

    def saveDataset(self):
        with open(self.path, "wb") as f:
            pickle.dump(self.examplesList, f)

    def createTrainAndValidationDataLoader(self, train_batch_size=64, validation_batch_size=64):
        graphList = [graph for example in self.examplesList for graph in example]
        graphs_train, graphs_validation = train_test_split(graphList, test_size=0.2, random_state=42)
        return (DataLoader(graphs_train, batch_size=train_batch_size),
                DataLoader(graphs_validation, batch_size=validation_batch_size))



















