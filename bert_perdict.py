

import os

import pandas as pd

from tensorflow import keras
from keras.models import Model
from sklearn.model_selection import train_test_split
from protein_bert.proteinbert import OutputType, OutputSpec, FinetuningModelGenerator, load_pretrained_model, finetune, evaluate_by_len
from protein_bert.proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs


def sample_perdict():
    BENCHMARKS_DIR = '/Users/lhwylp/Desktop/privatework/MLP_connect/protein_bert/protein_benchmarks/'
    # A local (non-global) binary output
    OUTPUT_TYPE = OutputType(False, 'binary')
    UNIQUE_LABELS = [0, 1]
    OUTPUT_SPEC = OutputSpec(OUTPUT_TYPE, UNIQUE_LABELS)

    # Loading the dataset
    train_set_file_path = os.path.join(BENCHMARKS_DIR, 'signalP_binary.train.csv')
    train_set = pd.read_csv(train_set_file_path).dropna().drop_duplicates()
    # 设置训练集和验证集的精确样本数量
    train_size = 20
    valid_size = 10
    # 随机抽样训练集和验证集
    train_set_sampled = train_set.sample(n=train_size + valid_size, random_state=0)

    # 划分训练集和验证集
    train_set, valid_set = train_test_split(train_set_sampled, stratify=train_set_sampled['label'], test_size=valid_size/(train_size + valid_size), random_state=0)


    test_set_file_path = os.path.join(BENCHMARKS_DIR, 'signalP_binary.test.csv')
    test_set = pd.read_csv(test_set_file_path).dropna().drop_duplicates()

    pretrained_model_generator, input_encoder = load_pretrained_model()
    model = pretrained_model_generator.create_model(seq_len=512)
    # 打印每一层的名称和类型
    for layer in model.layers:
        print(f"Layer Name: {layer.name}, Layer Type: {type(layer)}")
    # 假设使用倒数output前的global-merge2-norm-block6 可以随意换
    model_with_intermediate_output = Model(inputs=model.input, outputs=model.get_layer('global-merge2-norm-block6').output)
    layers_out = model_with_intermediate_output.predict(input_encoder.encode_X(valid_set['seq'].tolist(), 512))
    return layers_out