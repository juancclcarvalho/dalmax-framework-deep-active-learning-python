import time

# TensorFlow and Sklearn
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans 
from sklearn.pipeline import Pipeline
from scipy.spatial import distance
import numpy as np
# np.random.seed(43) # Definir a semente para o gerador de números aleatórios do NumPy

# Técnicas de Active Learning
class DalMaxSampler:
    @staticmethod
    def random_sampling(pool, batch_size):
        indices = np.random.choice(len(pool), batch_size, replace=False)
        return indices
    
    @staticmethod
    def diversity_sampling(pool, batch_size):
        start_time = time.time()
        print("Init diversity_sampling")

        # Garantir que o pool é bidimensional para o K-means
        flattened_pool = pool.reshape(len(pool), -1)
        print(f'Flattened pool shape: {flattened_pool.shape}')

        # Número de componentes para PCA e número de clusters para KMeans
        n_components = min(10, flattened_pool.shape[1]) # Ajuste para um número razoável de componentes. Esse número significa que o PCA irá reduzir a dimensionalidade para 10
        n_clusters = min(batch_size // 5, len(flattened_pool))  # Ajuste para um número razoável de clusters. Esse número significa que o K-means irá agrupar em 1/5 do tamanho do batch_size
        
        # Pipeline com PCA seguido de K-means
        pca = PCA(n_components=n_components)
        kmeans = KMeans(n_clusters=n_clusters, max_iter=200, random_state=42)
        predictor = Pipeline([('pca', pca), ('kmeans', kmeans)])

        # Ajuste no pool de dados achatados
        print("Fitting PCA and K-means")
        predictor.fit(flattened_pool)
        print("PCA and K-means fitted")

        # Transforme o pool com PCA para obter a representação reduzida
        reduced_pool = predictor.named_steps['pca'].transform(flattened_pool)

        # Obtenha os centros dos clusters a partir do espaço reduzido
        centers = predictor.named_steps['kmeans'].cluster_centers_
        
        # Encontra o índice da amostra mais próxima de cada centroide
        unique_indices = set()

        for center in centers:
            # Calcula a distância de cada ponto ao centroide atual
            distances = np.linalg.norm(reduced_pool - center, axis=1)
            closest_index = np.argmin(distances)
            unique_indices.add(closest_index)

        selected_indices = list(unique_indices)[:batch_size]
        
        end_time = time.time()
        print(f"Total time diversity_sampling: {end_time - start_time:.2f} seconds")

        return selected_indices

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
        start_time = time.time()
        print("Init core_set_selection")
        features = model.predict(pool)
        kmeans = KMeans(n_clusters=batch_size)
        print("Fitting K-means")
        kmeans.fit(features)
        centers = kmeans.cluster_centers_
        indices = []
        print("Finding closest samples to centers")
        for center in centers:
            idx = distance.cdist([center], features).argmin()
            indices.append(idx)

        end_time = time.time()
        print(f"Total time core_set_selection: {end_time - start_time:.2f} seconds")
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
