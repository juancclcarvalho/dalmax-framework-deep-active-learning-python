import time
import numpy as np
import os

# TensorFlow and Sklearn
import tensorflow as tf
import tensorflow_addons as tfa  # Certifique-se de que o TensorFlow Addons está instalado

from sklearn.cluster import KMeans
from scipy.spatial import distance
import numpy as np
# Definir a semente para o gerador de números aleatórios do NumPy
# np.random.seed(43)


# Técnicas de Active Learning
class DalMaxSampler:
    @staticmethod
    def random_sampling(pool, batch_size):
        indices = np.random.choice(len(pool), batch_size, replace=False)
        return indices
    @staticmethod
    def diversity_sampling_som(pool, batch_size, som_dim=(10, 10), epochs=100):
        start_time = time.time()
        print("Init diversity_sampling with Self-Organizing Maps (SOM)")

        flattened_pool = pool.reshape(len(pool), -1)
        input_dim = flattened_pool.shape[1]
        n_clusters = som_dim[0] * som_dim[1]

        # Inicializar pesos do SOM
        som_weights = tf.Variable(tf.random.normal([n_clusters, input_dim]), trainable=True)
        
        # Treinar o SOM
        for epoch in range(epochs):
            for data in flattened_pool:
                # Calcular distâncias e encontrar o melhor "matching unit" (BMU)
                data = tf.expand_dims(data, 0)
                distances = tf.reduce_sum((data - som_weights) ** 2, axis=1)
                bmu_index = tf.argmin(distances)
                
                # Atualizar pesos do BMU
                learning_rate = 0.5 * (1 - (epoch / epochs))
                som_weights[bmu_index].assign(som_weights[bmu_index] + learning_rate * (data - som_weights[bmu_index]))

        # Selecionar os índices das amostras mais próximas de cada cluster (BMU)
        selected_indices = []
        for weight in som_weights:
            distances = np.linalg.norm(flattened_pool - weight.numpy(), axis=1)
            closest_index = np.argmin(distances)
            selected_indices.append(closest_index)

        end_time = time.time()
        print(f"Total time with SOM: {end_time - start_time:.2f} seconds")
        return selected_indices[:batch_size]
    
    @staticmethod
    def diversity_sampling_mini_batch_kmeans(pool, batch_size, mini_batch_size=100):
        start_time = time.time()
        print("Init diversity_sampling with Mini-Batch K-Means")

        flattened_pool = pool.reshape(len(pool), -1)
        n_clusters = min(batch_size // 5, len(pool))

        # Definir o modelo Mini-Batch K-Means
        kmeans = tf.keras.layers.experimental.preprocessing.Discretization(num_bins=n_clusters)
        
        # Treinar o modelo em mini-batches
        for start in range(0, len(flattened_pool), mini_batch_size):
            end = start + mini_batch_size
            mini_batch = flattened_pool[start:end]
            kmeans.adapt(mini_batch)

        # Obter centros dos clusters e selecionar as amostras mais próximas
        centers = kmeans.get_weights()[0]
        selected_indices = []

        for center in centers:
            distances = np.linalg.norm(flattened_pool - center, axis=1)
            closest_index = np.argmin(distances)
            selected_indices.append(closest_index)

        end_time = time.time()
        print(f"Total time with Mini-Batch K-Means: {end_time - start_time:.2f} seconds")
        return selected_indices[:batch_size]

    @staticmethod
    def uncertainty_sampling(model, pool, batch_size):
        predictions = model.predict(pool)
        uncertainties = 1 - np.max(predictions, axis=1)
        return np.argsort(-uncertainties)[:batch_size]

    @staticmethod
    def query_by_committee(models, pool, batch_size):
        votes = np.array([model.predict(pool) for model in models])
        consensus = np.apply_along_axis(lambda x: np.bincount(x).argmax(), 0, votes.argmax(axis=2))
        disagreements = np.sum(votes.argmax(axis=2) != consensus, axis=0)
        return np.argsort(-disagreements)[:batch_size]

    # Core-Set Selection (K-Center)
    @staticmethod
    def core_set_selection(model, pool, batch_size):
        features = model.predict(pool)
        kmeans = KMeans(n_clusters=batch_size)
        kmeans.fit(features)
        centers = kmeans.cluster_centers_
        indices = []
        for center in centers:
            idx = distance.cdist([center], features).argmin()
            indices.append(idx)
        return indices

    # Adversarial Active Learning
    @staticmethod
    def adversarial_sampling(model, pool, batch_size):
        adv_images = []
        for img in pool:
            adv_img = tf.image.random_flip_left_right(img)  # Exemplo básico; personalize para ataques avançados
            adv_images.append(adv_img)
        adv_predictions = model.predict(np.array(adv_images))
        uncertainties = 1 - np.max(adv_predictions, axis=1)
        return np.argsort(-uncertainties)[:batch_size]
        
    import numpy as np

    @staticmethod
    def reinforcement_learning_sampling(agent, model, pool, batch_size):
        # Limite de imagens a processar
        limit = min(len(pool), 100)  # processa até 100 imagens ou menos se o pool for menor
        
        # Convertendo o pool de imagens para um formato 2D (número de imagens, dimensão da imagem)
        image_vectors = np.reshape(pool[:limit], (limit, agent.input_dim))

        # Selecionando ações para todas as imagens de uma vez
        actions = np.array([agent.select_action(image_vector) for image_vector in image_vectors])
        
        # Fazendo previsões para todas as imagens de uma vez
        predictions = model.predict(pool[:limit])  # Assume que o modelo pode lidar com batches

        # Calculando recompensas
        rewards = np.where(actions == 1, -np.max(predictions, axis=1), 0)

        # Empacotando recompensas com índices
        indexed_rewards = list(zip(rewards, range(limit)))

        # Ordenando as recompensas em ordem decrescente para selecionar as melhores
        indexed_rewards.sort(reverse=True, key=lambda x: x[0])

        # Retornando os índices das melhores imagens selecionadas
        return [idx for _, idx in indexed_rewards[:batch_size]]


    # Expected Model Change
    @staticmethod
    def expected_model_change(model, pool, batch_size):
        expected_changes = []
        for idx, img in enumerate(pool):
            prediction = model.predict(np.expand_dims(img, axis=0))
            expected_change = np.sum(np.abs(prediction - prediction.mean()))
            expected_changes.append((expected_change, idx))
        expected_changes.sort(reverse=True)
        return [idx for _, idx in expected_changes[:batch_size]]

    # Bayesian Active Learning
    @staticmethod
    def bayesian_sampling(model, pool, batch_size):
        predictions = [model.predict(pool, training=True) for _ in range(10)]
        predictions = np.array(predictions)
        predictive_variance = np.var(predictions, axis=0).sum(axis=1)
        return np.argsort(-predictive_variance)[:batch_size]
