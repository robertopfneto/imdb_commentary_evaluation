from keras import models, optimizers, layers
from main import NUM_WORDS


def build_model(input_data=NUM_WORDS):
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