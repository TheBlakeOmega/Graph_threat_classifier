from collections import Counter

from sklearn.metrics import accuracy_score
from torch.nn import Module, BCELoss
from torch_geometric.nn import GCNConv, Linear, global_mean_pool
from torch.nn.functional import sigmoid, relu, dropout
import torch
import pickle
import numpy as np


class GraphNetwork:

    def __init__(self):
        self.model = None

    def train(self, train_data_loader, validation_data_loader, params, device, save_path=None):
        self.model = GraphNeuralNetwork(100).to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=300)

        criterion = BCELoss()
        best_val_loss = np.inf
        best_val_acc = np.inf
        best_train_loss = 0
        best_train_acc = 0
        worst_loss_times = 0
        for epoch in range(10):

            # Train on batches
            total_acc = 0
            val_loss = 0
            val_acc = 0
            total_loss = 0
            self.model.train()
            for batch in train_data_loader:
                batch.to(device)
                optimizer.zero_grad()
                out = self.model(batch)
                out = out.squeeze()
                true_labels = batch.y.to(device)
                loss = criterion(out, true_labels)
                total_loss += float(loss)
                total_acc += accuracy_score(true_labels.tolist(), (out >= 0.5).tolist())
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                self.model.eval()
                for val_batch in validation_data_loader:
                    val_batch.to(device)
                    out = self.model(val_batch)
                    out = out.squeeze()
                    true_labels = val_batch.y.to(device)
                    loss = criterion(out, true_labels)
                    val_loss += float(loss)
                    val_acc += accuracy_score(true_labels.tolist(), (out >= 0.5).tolist())

            total_loss /= len(train_data_loader)
            total_acc /= len(train_data_loader)
            val_loss /= len(validation_data_loader)
            val_acc /= len(validation_data_loader)

            print(f'TRM: Epoch {epoch + 1:>3} '
                  f'| Train Loss: {total_loss:.3f} '
                  f'| Train Acc: {total_acc:.3f} '
                  f'| Val Loss: {val_loss:.3f} '
                  f'| Val Acc: {val_acc:.3f}')

            scheduler.step()

            if round(val_loss, 4) >= best_val_loss:
                worst_loss_times += 1
            else:
                worst_loss_times = 0
                best_val_loss = round(val_loss, 4)
                best_val_acc = round(val_acc, 4)
                best_train_loss = round(total_loss, 4)
                best_train_acc = round(total_acc, 4)
                # save best model's weights
                torch.save(self.model.state_dict(), 'tmp/temp_best_model_state_dict.pt')

            if worst_loss_times == 2:
                break

            # reload best model's weights
        self.model.load_state_dict(torch.load('tmp/temp_best_model_state_dict.pt', map_location=torch.device(device)))

        if save_path is not None:
            with open(save_path, 'wb') as file:
                print("TRM: Saving model in pickle object")
                pickle.dump(self.model, file)
                print("TRM: Model saved")
                file.close()

        scores = {
            'train_loss': best_train_loss,
            'train_accuracy': best_train_acc,
            'validation_loss': best_val_loss,
            'validation_accuracy': best_val_acc,
            'epochs': epoch + 1
        }

        return scores

    def test(self, test_example_list, device, prediction_type='soft'):
        real_labels = []
        predicted_labels = []
        self.model.eval()
        print("Examples to compute: " + str(len(test_example_list)))
        i = 0
        for test_example in test_example_list:
            real_labels.append(test_example[0].y.int().item())
            predicted_labels.append(self.predict_example(test_example, device, prediction_type=prediction_type))

            i += 1
            if i % 100 == 0:
                print(str(i) + ' test examples computed')
        return predicted_labels, real_labels

    def predict_example(self, test_example, device, prediction_type='soft'):
        predictions_on_sub_graphs = []
        for graph in test_example:
            graph.to(device)
            out = self.model(graph)
            out.squeeze(dim=0)
            predictions_on_sub_graphs.append(out.item())

        if prediction_type == 'soft':
            return 1 if np.mean(predictions_on_sub_graphs) > 0.5 else 0
        if prediction_type == 'hard':
            predictions_on_sub_graphs = [1 if prediction > 0.5 else 0 for prediction in predictions_on_sub_graphs]
            return 1 if max(predictions_on_sub_graphs, key=predictions_on_sub_graphs.count) > 0.5 else 0

    def loadModel(self, load_path, device):
        """
        Loads a model from storage
        :param load_path:
        path from which load the model
        :param device:
        device in which model will be used
        """
        with open(load_path, 'rb') as file:
            print("LM: Load model in pickle object")
            self.model = pickle.load(file)
            self.model.to(device)
            print("LM: Model loaded")
            file.close()


class GraphNeuralNetwork(Module):

    def __init__(self, num_node_features):
        super().__init__()
        torch.manual_seed(42)
        self.conv_layer = GCNConv(num_node_features, int(num_node_features / 2))
        self.lin = Linear(int(num_node_features / 2), 1)

    def forward(self, data):
        x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = self.conv_layer(x,
                            edge_index if edge_index.size(1) > 0 else None,
                            edge_weight=edge_weight if edge_weight.size(1) > 0 else None)
        x = relu(x)
        x = dropout(x, p=0.1)

        x = global_mean_pool(x, batch)

        x = self.lin(x)
        x = sigmoid(x)
        return x
