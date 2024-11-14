import os

import numpy as np
import random

import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout # type: ignore
from tensorflow.keras.applications import VGG16 # type: ignore

class DQNAgent:
    def __init__(self, input_dim, output_dim, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Modelo da Rede Neural
        self.model = self.build_model()
    
    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, input_dim=self.input_dim, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.output_dim, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
        return model
    
    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randint(0, 1)  # Escolha aleatória de 0 ou 1
        q_values = self.model.predict(np.expand_dims(state, axis=0), verbose=0)
        return np.argmax(q_values[0])  # Ação com maior valor Q

    def train(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            next_q_values = self.model.predict(np.expand_dims(next_state, axis=0), verbose=0)
            target += self.gamma * np.max(next_q_values[0])

        q_values = self.model.predict(np.expand_dims(state, axis=0), verbose=0)
        q_values[0][action] = target
        print(f"Training with state: {state}, action: {action}, reward: {reward}, next_state: {next_state}, done: {done}")
        self.model.fit(np.expand_dims(state, axis=0), q_values, verbose=1)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def create_parallel_model(input_shape, num_classes):
    # Estratégia de distribuição
    strategy = tf.distribute.MirroredStrategy()
    
    with strategy.scope():
        model = create_model(input_shape, num_classes, mult_gpu=True)
    return model

# Função para criar um modelo simples
def create_model(input_shape, num_classes, mult_gpu=False, use_gpu=0):

    if mult_gpu is False:
        # Defina qual GPU será visível antes de importar o TensorFlow
        os.environ["CUDA_VISIBLE_DEVICES"] = str(use_gpu)

    print(f"\n\n--------------------------------------")
    print(f"CREATING MODEL")
    print(f"TensorFlow version: {tf.__version__}")
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print(f"Using GPU: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"Input shape: {input_shape}")
    print(f"Number of classes: {num_classes}")
    print(f"--------------------------------------\n\n")

    # base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    # Base model Resnet 50
    base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

    # base_model = EfficientNetB7(weights='imagenet', include_top=False, input_shape=input_shape)
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomContrast(0.2),
    ])

    model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dropout(0.4),
            Dense(num_classes, activation='softmax')
        ])
    
    # Anotação: categorical_crossentropy é a função de perda correta para classificação multiclasse
    # sparse_categorical_crossentropy é a função de perda correta para classificação multiclasse com rótulos inteiros
    

    # Aumentar learning rate: 0.1, 0.01, 0.001: testar
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0007)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()
    return model