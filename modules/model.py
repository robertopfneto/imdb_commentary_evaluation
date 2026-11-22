from keras import models, optimizers, layers
import re # expressao regular
from typing import List
import numpy as np

from modules.vetorization import vet_sequence


def build_model(input_data):
    model = models.Sequential([
        # recebe os valores de 10.000 posições e vai combinar seus valores com 16 neuronios com a função
        layers.Dense(16, activation="relu", input_shape=(input_data, )),  # relu remove os negativos
        layers.Dense(16, activation="relu"),  # relu remove os negativos
        # output
        layers.Dense(1, activation="sigmoid")
    ])
    
    model.compile(
        optimizer=optimizers.RMSprop(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    
    return model


# Funções para classificar texto livre

# PREPARAÇÃO
# - Limpar o texto
# - Separar palavras
# -  Procurar os codigos no dicionario de treinamento
# - gerar o vetor de presença

# REMOVE O ACENTO E PONTUAÇÃO
# o retorno p (predição) e 1 (classificação)

# Define a limpeza simplificada de uma frase em inglês e devolve uma lista de palavras.


def preprocess_text(text: str) -> List[str]:
    # Converte letras para minúsculas: Movie e movie passam a ter a mesma escrita.
    text = text.lower()
    # Substitui endereços de sites por espaços.
    text = re.sub(r"http\S+|www\S+", " ", text)
    # Substitui menções iniciadas por @ por espaços.
    text = re.sub(r"@\S+", " ", text)
    # Mantém apenas a-z e espaços; também remove pontuação, apóstrofos, dígitos e acentos.
    text = re.sub(r"[^a-z ]", " ", text)
    # Reduz espaços repetidos e remove espaços do início e do fim do texto.
    text = re.sub(r"\s+", " ", text).strip()
    # Separa a frase em palavras usando os espaços.
    tokens = text.split(" ")
    # Retorna apenas os elementos não vazios, em uma única lista.
    return [t for t in tokens if t]
# Define a conversão das palavras em códigos numéricos do vocabulário IMDB.



def text_to_imdb_indices(tokens: List[str], # Recebe o dicionário que relaciona cada palavra ao seu índice.
                         word_index: dict, # Usa por padrão o mesmo limite de vocabulário do treinamento.
                         num_words):
    
    # Cria a lista que receberá os códigos desta frase.
    indices = []
    # Percorre cada palavra produzida pela limpeza do texto.
    for token in tokens:
        # Verifica se a palavra existe no dicionário completo.
        if token in word_index:
            # Soma 3 para compatibilizar o índice com os códigos reservados do IMDB.
            idx = word_index[token] + 3
        # Trata o caso em que a palavra não existe no dicionário.
        else:
            # Usa 2 como marcador de palavra desconhecida; não representa sentimento negativo.
            idx = 2
        # Mantém somente códigos abaixo do limite; palavras conhecidas acima dele são ignoradas.
        if idx < num_words:
            # Acrescenta o código aceito à lista desta frase.
            indices.append(idx)
    # Retorna uma lista de códigos; se ficar vazia, retorna apenas o marcador 2.
    return indices or [2]
# Define a preparação completa de uma frase para a entrada da rede.


def vectorize_text(text: str, word_index: dict, num_words: int) -> np.ndarray:
    # Aplica a limpeza e separa as palavras.
    tokens = preprocess_text(text)
    # Converte as palavras em índices usando o dicionário do treinamento.
    indices = text_to_imdb_indices(tokens, word_index, num_words=num_words)
    # Retorna uma matriz com uma linha e 10.000 colunas de presença de palavras.
    return vet_sequence([indices],dimension=num_words)


# Define a classificação da frase; threshold é o limiar, por padrão 0,5.
def inference(text: str, model: dict, work_index: dict, num_words: int, threshold: float=0.5):
    # Prepara a representação numérica da frase, sem informar um gabarito à rede.
    x = vectorize_text(text, work_index, num_words=num_words)
    # Prevê sem treinar e extrai a estimativa da única linha e da única saída.
    proba = model.predict(x, verbose=0)[0, 0]
    # Usa positivo para estimativa maior ou igual a 0,5 e negativo para valor menor.
    label = "POSITIVE" if proba >= threshold else "NEGATIVE"
    # Retorna a tupla já usada no exemplo: estimativa e rótulo, para desempacotar em p e l.
    return proba, label 