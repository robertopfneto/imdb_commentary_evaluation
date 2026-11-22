
from keras.callbacks import ModelCheckpoint

# Dos 25k exemplos iniciais, 15k sao para treinamento e 10k para validacao
# epoca, batch = 512 
# Early Stopping: quantas épocas o modelo vai aguardar sem ter nenhuma melhora

def train(partial_x_train, partial_y_train, x_val, y_val, callbacks, model, epochs, batch_size):
    history = model.fit(
        # fornece os 15k comentarios do treino -> partial_x_train
        # e os routolos -> partial_y_train
        partial_x_train, partial_y_train,
        epochs=epochs,
        batch_size=batch_size,
        
        # usa validação para acompanhar o desempenho,
        validation_data = (x_val, y_val),
        callbacks=callbacks,
        # exibe o progresso e as medidas de cada epoca
        verbose=1  
    )

    return history