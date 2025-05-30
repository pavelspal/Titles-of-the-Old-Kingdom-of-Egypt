import torch
import torch.nn as nn
import json
import numpy as np
import pandas as pd

try:
    from supp.support_load import read_csv
except:
    from support_load import read_csv


# Define the Neural Network
class BinaryClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes, activation_name=None, dropout_prob=0, input_dropout_prob=0):
        super(BinaryClassifier, self).__init__()
        hidden_sizes = hidden_sizes if isinstance(hidden_sizes, list) else [hidden_sizes]
        dropout_prob = dropout_prob if isinstance(dropout_prob, list) else [dropout_prob] * len(hidden_sizes)
        activation_name = activation_name if isinstance(activation_name, list) else [activation_name] * len(hidden_sizes)

        # Map string name to activation function class
        activation_functions = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid()
        }

        layers = []
        previous_size = input_size
        if input_dropout_prob != 0:
            layers.append(nn.Dropout(input_dropout_prob))
        for hidden_size, do, act in zip(hidden_sizes, dropout_prob, activation_name):
            layers.append(nn.Linear(previous_size, hidden_size))
            activation = activation_functions.get(act, nn.ReLU())
            layers.append(activation)
            layers.append(nn.Dropout(do))  # Add dropout
            previous_size = hidden_size
        layers.append(nn.Linear(previous_size, 1))
        self.model = nn.Sequential(*layers)

        self._initialize_biases()

    # For each nn.Linear layer, set 0.01
    def _initialize_biases(self):
        for module in self.model:
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.fill_(0.01)

    def forward(self, x):
        return self.model(x)


# Define function to save NN model architecture
def save_nn_info(input_size, hidden_size, activation_name, features, path):
    # Check that hidden_size is list
    hidden_size = hidden_size if isinstance(hidden_size, list) else [hidden_size]
    # Define dict with info
    dict_info = {'input_size': input_size,
                 'hidden_size': hidden_size,
                 'activation_name': activation_name,
                 'features': features
                 }

    # Save dict with info about model
    with open(f"{path}.json".replace('.pth', ''), "w") as f:
        json.dump(dict_info, f, indent=4)


# Define function to load NN model architecture
def load_nn_info(path):
    # Load dict
    with open(path.replace('.pth', '.json'), "r") as f:
        loaded_dict = json.load(f)

    return loaded_dict


# Define function to load NN model
def load_nn_model(path):
    # Load model info
    dict_info = load_nn_info(path)
    # Instantiate model
    model = BinaryClassifier(dict_info['input_size'],
                             dict_info['hidden_size'],
                             dict_info['activation_name'])
    # Load weights
    model.load_state_dict(torch.load(path))

    # Get feature names
    features_names = dict_info['features']

    # Print model architecture
    print(f'MODEL ARCHITECTURE:\n{model}\n\n')

    return model, features_names


# Define function that can be easy use for model prediction
def convert_model_nn_to_forward(model, feature_names):
    def model_forward(x):
        # if passed pandas dataframe, convert it to numpy
        if isinstance(x, pd.DataFrame):
            x = x.loc[:, feature_names]

        x = np.array(x)
        x = torch.tensor(x, dtype=torch.float32)
        model.eval()
        with torch.no_grad():  # Ensure gradients are not computed
            output = model(x)
            output = torch.sigmoid(output)  # Apply sigmoid
        return output.numpy()  # Convert to NumPy

    return model_forward, feature_names


# Load model from path
def get_model_nn_forward(path):
    model, feature_names = load_nn_model(path)
    return convert_model_nn_to_forward(model, feature_names)

