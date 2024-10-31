import time

# TensorFlow and Sklearn
import tensorflow as tf
from sklearn.cluster import KMeans
from scipy.spatial import distance
import numpy as np

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
            # Imprimir o shape de pool
            print(f'Pool shape: {pool.shape}')
            # Achata cada amostra no pool para garantir uma entrada bidimensional para o K-means
            flattened_pool = pool.reshape(len(pool), -1)
            print(f'Flattened pool shape: {flattened_pool.shape}')
            
            # Realiza o clustering no pool de dados achatados com o número de clusters igual ao batch_size
            kmeans = KMeans(n_clusters=batch_size, random_state=42)
            kmeans.fit(flattened_pool)
            
            # Encontra os índices dos pontos mais próximos de cada centroide do cluster
            centers = kmeans.cluster_centers_
            indices = []
            
            for center in centers:
                # Calcula a distância de cada ponto ao centroide atual
                distances = np.linalg.norm(flattened_pool - center, axis=1)
                # Seleciona o índice do ponto mais próximo do centroide
                closest_index = np.argmin(distances)
                indices.append(closest_index)
            
            end_time = time.time()
            
            text_time = f"Total time: {end_time - start_time:.2f} seconds"
            print(text_time)

            return indices

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
