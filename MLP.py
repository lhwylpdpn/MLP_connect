import time

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Model

import numpy as np
import pandas as pd
import os

from Bio import SeqIO
from Bio.PDB.PDBParser import PDBParser
from PGAT_ABPp.predict.GraphFromPDB import graph_from_pdb, prepare_batch, Dataset
from PGAT_ABPp.predict.Model import MultiHeadGraphAttention, TransformerEncoderReadout, GraphAttentionNetwork

import tqdm
import warnings

warnings.filterwarnings("ignore")
np.random.seed(42)
tf.random.set_seed(42)

# Data Preprocess
df = pd.read_csv("./PGAT_ABPp/predict/example_data/data.csv")  # path to your CSV file
seq_length = [len(i) for i in df.seq.values]
U50_embeddings = np.load('./PGAT_ABPp/predict/example_data/data_U50.npy')  # path to your NPY file
U50_embeddings_list = []
start_idx = 0
for length in seq_length:
    end_idx = start_idx + length
    protein_slice = U50_embeddings[start_idx:end_idx, :]
    U50_embeddings_list.append(protein_slice)
    start_idx = end_idx

# Dataset
x_pred = graph_from_pdb(df, U50_embeddings_list)
y_pred = df.label
pred_dataset = Dataset(x_pred, y_pred)

# Define hyper-parameters
HIDDEN_UNITS = 10
NUM_HEADS = 6
NUM_LAYERS = 1
BATCH_SIZE = 32

# Build model
gat_model = GraphAttentionNetwork(1024, HIDDEN_UNITS, NUM_HEADS, NUM_LAYERS, BATCH_SIZE)


def predict_with_intermediate(model, dataset):
    all_predictions = []
    all_intermediate_outputs = []

    for batch in dataset:  # 迭代数据集

        x, labels = batch
        x = [tf.cast(x[0], tf.float32), tf.cast(x[1], tf.float32), x[2]]  # 假设 x 是一个列表，并且前两个元素需要转换
        node_features, molecule_indicator, pair_indices = x  # 正确解包

        predictions, intermediate_outputs = predict_batch(model, [node_features, pair_indices, molecule_indicator])
        all_predictions.append(predictions)
        all_intermediate_outputs.append(intermediate_outputs)

    return np.concatenate(all_predictions), all_intermediate_outputs


def predict_batch(model, inputs):
    intermediate_outputs = []
    x = model.layers[1](inputs[0])  # preprocess layer

    for i in range(2, len(model.layers) - 2, 2):  # attention layers
        x = model.layers[i]([x, inputs[1]]) + x
        intermediate_outputs.append(x)
    x = model.layers[-2]([x, inputs[2]])  # readout layer
    intermediate_outputs.append(x)
    x = model.layers[-1](x)  # output layer
    return x, intermediate_outputs


# Load Model
gat_model.load_weights('./PGAT_ABPp/predict/pgat_abpp.h5')
readout_layer = gat_model.get_layer("transformer_encoder_readout")

model_with_multiple_outputs = Model(inputs=gat_model.input, outputs=readout_layer.output)

# Predict
layers_out = model_with_multiple_outputs.predict(pred_dataset)
#
# y = (predictions > 0.5).astype(int)
# y =  np.ravel(y)
# print('Prediction results:', y)