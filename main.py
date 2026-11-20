import numpy as np
import tensorflow as tf # fornece os calculos que a rede neural precisa
from keras.datasets import imdb # dataset

from classification_text import classify_text
from model import build_model
from vetorization import vet_sequence
from keras import callbacks



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



# vetoriza o dataset
x_train = vet_sequence(train_data)
x_test = vet_sequence(test_data)

# converte os rotulos para numeros
y_train = np.asarray(train_labels, dtype="float32")
y_test = np.asarray(test_labels, dtype="float32")


model = build_model()

### Separar o conjundo de validação e Early Stopping

# Dos 25k exemplos iniciais, 15k sao para treinamento e 10k para validacao
# epoca, batch = 512 
# Early Stopping: quantas épocas o modelo vai aguardar sem ter nenhuma melhora

VAL_SIZE = 10000
x_val, partial_x_train = x_train[:VAL_SIZE], x_train[VAL_SIZE:]

y_val, partial_y_train = y_train[:VAL_SIZE], y_train[VAL_SIZE:]

early_stopping = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    # fornece os 15k comentarios do treino -> partial_x_train
    # e os routolos -> partial_y_train
    partial_x_train, partial_y_train,
    epochs=20,
    batch_size=512,
    
    # usa validação para acompanhar o desempenho,
    validation_data = (x_val, y_val),
    callbacks=[early_stopping],
    
    # exibe o progresso e as medidas de cada epoca
    verbose=1  
)


### TESTAR MODELO

# Classificar uma frase em ingles nova com a rede ja treinada
# p = estimativa(0..1)
# l = rótulo
p, l = classify_text("I should have gone to see the Pelé movie")
print(p, l)

p, l = classify_text("It's the best movie ever")
print(p, l)