import torch
import traceback
from datetime import datetime
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import utils
from dataset import JsonDataset, GraphDataset
from embeddingModel import EmbeddingModel
from graphConvolutionalNetworkClassifier import GraphNetwork
import matplotlib.pyplot as plt


class PipeLineManager:

    def __init__(self, configuration, dataset_configuration, result_file):
        self.configuration = configuration
        self.ds_configuration = dataset_configuration
        self.result_file = result_file
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def runPipeline(self):
        """
        This method runs the pipeline chosen according to the configuration file
        """
        print("SELECTED PIPELINE: " + self.configuration['chosen_pipeline'])
        if self.configuration['chosen_pipeline'] == 'PREPROCESS_JSON_DATASET':
            try:
                self._preProcessJSONDataset()
                print("JSON dataset preprocessed")
            except Exception as e:
                traceback.print_exc()
                raise Exception("Error during JSON dataset preprocessing")
        elif self.configuration['chosen_pipeline'] == 'TRAIN_WORD2VEC_MODEL':
            try:
                self._trainWord2VecModel()
                print("Word2Vec generated")
            except Exception as e:
                traceback.print_exc()
                raise Exception("Error during Word2Vec model training")
        elif self.configuration['chosen_pipeline'] == 'CREATE_GRAPH_DATASET':
            try:
                self._generateGraphDataset()
                print("Graph dataset generated")
            except Exception as e:
                traceback.print_exc()
                raise Exception("Error during Graph dataset creation")
        elif self.configuration['chosen_pipeline'] == 'TRAIN_GNN_MODEL':
            try:
                self._trainGNNModel()
                print("GNN model trained")
            except Exception as e:
                traceback.print_exc()
                raise Exception("Error during GNN training")
        elif self.configuration['chosen_pipeline'] == 'TEST_GNN_MODEL':
            try:
                self._testGNNModel()
                print("GNN model trained")
            except Exception as e:
                traceback.print_exc()
                raise Exception("Error during GNN testing")

    def _preProcessJSONDataset(self):
        allowed_packages = [
            "Landroid/accounts",
            "Landroid/app",
            "Landroid/bluetooth",
            "Landroid/content",
            "Landroid/location",
            "Landroid/media",
            "Landroid/net",
            "Landroid/nfc",
            "Landroid/provider",
            "Landroid/telecom",
            "Landroid/telephony",
        ]

        train_dataset = JsonDataset(self.ds_configuration['jsonTrainPathDataset'])
        print("Filtering train dataset by allowed packages")
        train_dataset.filterDatasetByPackages(allowed_packages)
        print("Computing TF-IDF")
        top_tfidf_string = train_dataset.compute_tf_idf(int(self.configuration['top_n_tf_idf_strings']),
                                                        "top_" + self.configuration['top_n_tf_idf_strings']
                                                        + "_tdf_idf_strings.pkl")
        print("Filtering train dataset by TFIDF top strings")
        train_dataset.filterDatasetByTFIDF(top_tfidf_string)
        print("Deleting files with empty properties from train dataset")
        train_dataset.delete_files_with_empty_properties()

        test_dataset = JsonDataset(self.ds_configuration['jsonTestPathDataset'])
        print("Filtering test dataset by allowed packages")
        test_dataset.filterDatasetByPackages(allowed_packages)
        print("Filtering test dataset by TFIDF top strings")
        test_dataset.filterDatasetByTFIDF(top_tfidf_string)
        print("Deleting files with empty properties from test dataset")
        test_dataset.delete_files_with_empty_properties()

    def _trainWord2VecModel(self):
        print("Extracting sequences from dataset in " + self.ds_configuration['jsonTrainPathDataset'])
        train_dataset = JsonDataset(self.ds_configuration['jsonTrainPathDataset'])
        extracted_sequences = train_dataset.extract_sequences()
        print("Training Word2Vec model")
        embedding_model = EmbeddingModel()
        embedding_model.train(extracted_sequences)
        print("Saving Word2Vec model")
        embedding_model.saveModel()
        print(embedding_model.stringEmbeddingModel.wv)

    def _generateGraphDataset(self):
        print("Loading Word2Vec model")
        embedding_model = EmbeddingModel()
        embedding_model.loadModel()

        print("Generating train graph dataset")
        train_json_dataset = JsonDataset(self.ds_configuration['jsonTrainPathDataset'])
        train_graph_dataset = GraphDataset(self.ds_configuration['graphTrainDatasetPath'])
        train_graph_dataset.generateGraphDataset(train_json_dataset, embedding_model)
        print("Saving generated train dataset")
        train_graph_dataset.saveDataset()

        print("Generating test graph dataset")
        test_json_dataset = JsonDataset(self.ds_configuration['jsonTestPathDataset'])
        test_graph_dataset = GraphDataset(self.ds_configuration['graphTestDatasetPath'])
        test_graph_dataset.generateGraphDataset(test_json_dataset, embedding_model)
        print("Saving generated test dataset")
        test_graph_dataset.saveDataset()

    def _trainGNNModel(self):
        print("Loading graph dataset")
        train_graph_dataset = GraphDataset(self.ds_configuration['graphTrainDatasetPath'])
        train_graph_dataset.loadDataset()
        train_data_loader, validation_data_loader = train_graph_dataset.createTrainAndValidationDataLoader()

        model = GraphNetwork()
        save_path = self.configuration['path_models'] + "/trainedGCN.pkl"
        print("Starting GCN training on " + torch.cuda.get_device_name(0) + " " + str(self.device))
        start_train_time = np.datetime64(datetime.now())
        scores = model.train(train_data_loader, validation_data_loader, {}, self.device, save_path)
        end_train_time = np.datetime64(datetime.now())
        self.result_file.write("Trained model result:" + "\nLoss: " + str(scores['train_loss']) +
                               "\nAccuracy: " + str(scores['train_accuracy']) +
                               "\nVal Loss: " + str(scores['validation_loss']) +
                               "\nVal Accuracy: " + str(scores['validation_accuracy']) +
                               "\nTrain time: " + str(utils.convertTimeDelta(start_train_time, end_train_time)) + "\n")

    def _testGNNModel(self):
        print("Loading graph dataset")
        test_graph_dataset = GraphDataset(self.ds_configuration['graphTestDatasetPath'])
        test_graph_dataset.loadDataset()

        print("Loading GNN model")
        model = GraphNetwork()
        load_path = self.configuration['path_models'] + "/trainedGCN.pkl"
        model.loadModel(load_path, self.device)
        start_test_time = np.datetime64(datetime.now())
        predicted_labels, real_labels = model.test(test_graph_dataset.examplesList, self.device,
                                                   self.configuration['prediction_type'])
        print(predicted_labels)
        print(real_labels)
        end_test_time = np.datetime64(datetime.now())
        print("Computing confusion matrix")
        labels = list(set(real_labels))
        test_confusion_matrix = confusion_matrix(real_labels, predicted_labels, labels=labels)
        print("Computing classification report")
        test_classification_report = classification_report(real_labels, predicted_labels, digits=3)
        self.result_file.write("Test confusion matrix:\n")
        self.result_file.write(str(test_confusion_matrix) + "\n")
        self.result_file.write("Test classification report:\n")
        self.result_file.write(str(test_classification_report) + "\n")
        self.result_file.write("Test computation time:\n")
        self.result_file.write(str(end_test_time - start_test_time) + "\n")
        test_confusion_matrix_plot = ConfusionMatrixDisplay(test_confusion_matrix, display_labels=labels)
        test_confusion_matrix_plot.plot()
        plt.title("Test Confusion Matrix " + self.configuration['prediction_type'])
        plt.savefig("test_confusion_matrix.png")
        plt.close()









































