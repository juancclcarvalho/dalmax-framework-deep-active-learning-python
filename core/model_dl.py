import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import VGG16

# Imprimir a versão do TensorFlow
print(f"TensorFlow version: {tf.__version__}")
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

def create_parallel_model(input_shape, num_classes):
     # Construir o modelo dentro do escopo da estratégia de distribuição
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = create_model(input_shape, num_classes)
    return model

# Função para criar um modelo simples
def create_model(input_shape, num_classes):
   
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    
    model = Sequential([
            base_model,
            Flatten(),
            Dense(256, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy','mae', 'mse']) #  
    return model