from bert_perdict import sample_perdict as bert_layer
from PGAT_perdict import sample_perdict as pgat_layer
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Concatenate
from keras.optimizers import Adam

def get_layer_out():
    bert_output = bert_layer()
    pgat_output = pgat_layer()
    return bert_output, pgat_output


def create_MLP_with_layer():
    bert_output, pgat_output=get_layer_out()
    input1 = Input(shape=(bert_output.shape[1],))
    input2 = Input(shape=(pgat_output.shape[1],))
    hidden_layer1=Dense(128,activation='relu')(input1)
    hidden_layer2=Dense(64,activation='relu')(input2)
    combined = Concatenate()([hidden_layer1, hidden_layer2])
    hidden_layer3 = Dense(64, activation='relu')(combined)
    output_layer = Dense(1, activation='sigmoid')(hidden_layer3)  # 二分类
    mlp_model = Model(inputs=[input1, input2], outputs=output_layer)
    mlp_model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    mlp_model.summary()
    return mlp_model

if __name__ == '__main__':
    model=create_MLP_with_layer()
    y_train = np.random.randint(0, 2, size=(10, 1))  # 示例二分类标签
    output1,output2=get_layer_out()
    model.fit([output1, output2], y_train, epochs=10, batch_size=2)

    y=model.predict([output1, output2])
    #
    y = (y > 0.5).astype(int)
    y =  np.ravel(y)
    print('Prediction results:', y)