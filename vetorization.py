import numpy as np
from main import NUM_WORDS

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