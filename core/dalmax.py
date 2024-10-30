import random
import numpy as np

# TensorFlow and Sklearn
import tensorflow as tf
from sklearn.cluster import KMeans
from scipy.spatial import distance

# Técnicas de Active Learning
class DalMaxSampler:
    @staticmethod
    def random_sampling(pool):
        return random.choice(pool)
    
    @staticmethod
    def diversity_sampling(pool, batch_size):
        indices = np.random.choice(len(pool), batch_size, replace=False)
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

    # Reinforcement Learning for Active Learning
    @staticmethod
    def reinforcement_learning_sampling(agent, model, pool, batch_size):
        rewards = []
        for idx, img in enumerate(pool):
            action = agent.select_action(img)
            prediction = model.predict(np.expand_dims(img, axis=0))
            reward = -np.max(prediction) if action == 1 else 0
            rewards.append((reward, idx))
        rewards.sort(reverse=True)
        return [idx for _, idx in rewards[:batch_size]]

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
