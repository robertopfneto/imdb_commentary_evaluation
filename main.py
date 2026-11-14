import numpy as np, random, os # vetores, matrizes, recursos
import tensorflow as tf # fornece os calculos que a rede neural precisa
from keras.datasets import imdb # dataset
from keras import models, optimizers, callbacks, layers

import re # expressao regular
from typing import List


print("Tensorflow: ", tf.__version__)
print("Numpy: ", np.__version__)

### Carregar o dataset

NUM_WORDS = 10000 # Limitar os indices de 10.000: vocabulários frequentes

# rotulos: 0 - negativo, 1 - positivo. 

# carrega comentarios   
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=NUM_WORDS)
print(f"Tamanho do treino: {len(train_data)}")
print(f"Tamanho do teste: {len(test_labels)}")

# vetorização numerica dos comentarios
work_index = imdb.get_word_index() # carrega um dicionario de palavras
print(f"Palavra index -> index: {list(work_index.items())[:10]}")

# reversa Index -> palavra
reverse_word_index = {idx: word for word, idx in work_index.items()}
print(f"Indice -> palavra: {list(reverse_word_index.items())[:10]}")

# cada comentario vira uma linha com 10k de posições
# a posição de uma palavra recebe: 1 - se estiver presente e 0 - se estiver ausente
# Valores atribuidos na entrada e nao nos rotulos de sentimento; Se uma palavra se repete nao aumenta o valor   

def vet_sequence(sequence, dimension=NUM_WORDS):
    # cria uma linha por comentario e coluna por indice, inicialmente com zero
    results = np.zeros((len(sequence), dimension), dtype="float32")      
    for row, seq in enumerate(sequence):
        seq = [i for i in seq if i < dimension]
        results[row, seq] = 1 
    return results

# vetoriza o dataset
x_train = vet_sequence(train_data)
x_test = vet_sequence(test_data)

# converte os rotulos para numeros
y_train = np.asarray(train_labels, dtype="float32")
y_test = np.asarray(test_labels, dtype="float32")


## Configurações do modelo

def build_model(input_data=NUM_WORDS):
    model = models.Sequential([
        # recebe os valores de 10.000 posições e vai combinar seus valores com 16 neuronios com a função
        layers.Dense(16, activation="relu", input_shape=(input_data, )),  # relu remove os negativos
        layers.Dense(16, activation="relu"),  # relu remove os negativos
        # output
        layers.dense(1, activation="sigmoid")
    ])
    
    model.compile(
        optimizer=optimizers.RMSprop(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    
    return model