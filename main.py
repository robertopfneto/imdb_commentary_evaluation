from keras import callbacks
import numpy as np
import os

import tensorflow as tf # fornece os calculos que a rede neural precisa
from keras.datasets import imdb # dataset

from modules.train_model import train
from modules.model import build_model, inference
from modules.vetorization import vet_sequence
from keras.callbacks import ModelCheckpoint


print("Tensorflow: ", tf.__version__)
print("Numpy: ", np.__version__)

### Carregar o dataset

NUM_WORDS = 10000 # Limitar os indices de 10.000: vocabulários frequentes
VAL_SIZE = 10000
MODEL_PATH = "text_model.keras"
EPOCHS = 20
BATCH_SIZE = 512
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
x_train = vet_sequence(train_data, dimension=NUM_WORDS)
x_test = vet_sequence(test_data, dimension=NUM_WORDS)

# converte os rotulos para numeros
y_train = np.asarray(train_labels, dtype="float32")
y_test = np.asarray(test_labels, dtype="float32")

model = build_model(input_data=NUM_WORDS)

## chamando treino

if not os.path.exists(MODEL_PATH):
    print("Pesos nao encontrados, iniciando treino...")
    
    x_val, partial_x_train = x_train[:VAL_SIZE], x_train[VAL_SIZE:]
    y_val, partial_y_train = y_train[:VAL_SIZE], y_train[VAL_SIZE:]

    early_stopping = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )

    checkpont_cb = ModelCheckpoint(
        filepath=MODEL_PATH,
        save_best_only=True, # salva o melhor modelo entre as epocas
        mode="min", # min para loss e max para acuracia
        verbose=1
    )

    callbacks = [early_stopping, checkpont_cb]
    history = train(partial_x_train, partial_y_train, x_val, y_val, callbacks, model, epochs=EPOCHS, batch_size=BATCH_SIZE)

else:
    print(f"Pesos encontrados em {MODEL_PATH}")
    model = model.load_weights(MODEL_PATH)



### TESTAR MODELO

# Classificar uma frase em ingles nova com a rede ja treinada
# p = estimativa(0..1) | p = predict
# l = rótulo | l = label
p, l = inference("I should have gone to see the Pelé movie", model=model, work_index=work_index, num_words=NUM_WORDS)
print(p, l)

p, l = inference("It's the best movie ever", model=model, work_index=work_index,  num_words=NUM_WORDS)
print(p, l)


## ATIVIDADE
