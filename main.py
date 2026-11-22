from keras import callbacks
import numpy as np
import os

import tensorflow as tf # fornece os calculos que a rede neural precisa
from keras.datasets import imdb # dataset

from modules.train_model import train
from modules.model import build_model, inference
from modules.vetorization import vet_sequence
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

import json

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

## chamando treino

if not os.path.exists(MODEL_PATH):
    print("Pesos nao encontrados, iniciando treino...")
    model = build_model(input_data=NUM_WORDS)

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
    model = load_model(MODEL_PATH)



### TESTAR MODELO

# Classificar uma frase em ingles nova com a rede ja treinada
# p = estimativa(0..1) | p = predict
# l = rótulo | l = label


# atividade analise

# abre o jso com os comentarios.json
with open('comentarios.json', 'r') as f:
    json_file = dict(json.load(f))

# carrega os comentarios em uma lista
lista_comentarios = [c for c in json_file.values()]


# lista_comentarios[indice_label, indice_comentario]
# 1 - positivo
# 0 - negativo

labels_positivas = lista_comentarios[1]
labels_negativas = lista_comentarios[0]

# contando verdadeiros positivos, verdadeiros negativos, falsos positivos e falsos negativos

TP, TN, FP, FN = 0, 0, 0, 0

lista_resultados = []

for frase in labels_positivas:
    p, l = inference(frase, model=model, work_index=work_index,  num_words=NUM_WORDS)

    if l == "NEGATIVE":
       lista_resultados.append((p, l, "FN"))
       FN+=1
        
    elif l == "POSITIVE":
       lista_resultados.append((p, l, "TP"))
       TP+=1

for frase in labels_negativas:
    p, l = inference(frase, model=model, work_index=work_index,  num_words=NUM_WORDS)

    if l == "NEGATIVE":
       lista_resultados.append((p, l, "TN"))
       TN+=1
    elif l == "POSITIVE":
       FN+=1
       lista_resultados.append((p, l, "FP"))


# exibindo resultados
print("| Comentario | Sentimento Real | Sentimento Predito | Tipo Resultado | Predict |")
for i, frase in enumerate(labels_negativas):
    print(f"{frase} | NEGATIVE | {lista_resultados[i][1]} | {lista_resultados[i][2]} | {lista_resultados[i][0]}")

for i, frase in enumerate(labels_positivas):
    print(f"{frase} | POSITIVE | {lista_resultados[i][1]} | {lista_resultados[i][2]} | {lista_resultados[i][0]}")

total = TP+TN+FP+FN
print("Total Contabilizados:" +str(total))
print(f"TP: {TP} ")
print(f"TN: {TN} ")
print(f"FP: {FP} ")
print(f"FN: {FN} ")


# calculando métricas
precisao = float(TP / (TP + FP))
recall = float(TP / (TP + FN))
f1_score = float((2*precisao*recall) / (precisao+recall))
acuracia = (TP + TN )/( TP + FN + TN + FP)
print()
print("Precisao: " +str(precisao))
print("Recall: " +str(recall))
print("F1-Score: " +str(f1_score))
print("Acurácia: " +str(acuracia))

"""
ANALISE DOS RESULTADOS

Em todas que o modelo marcou como positiva ele acertou, eh a justificativa para ter 1.0 de precisão,
entretanto, deixou passar um positivo das 5 positivas, por isso decaiu 20% na metrica de recall que ficou nas 80%,
O f1-score é uma média harmônica entre os dois casos, onde conseguimos ver que em geral, o modelo reflete uma boa capacidade
de classificar as frases, confirmado pela acurácia de 0.9

"""