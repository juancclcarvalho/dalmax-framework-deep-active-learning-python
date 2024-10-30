import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
# Salvar o relatório como PDF
from matplotlib.backends.backend_pdf import PdfPages

# Carregar e preparar o CIFAR-10
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalização

# Definir o modelo (ResNet18 como exemplo)
base_model = tf.keras.applications.ResNet50(weights=None, input_shape=(32, 32, 3), classes=10)
model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# Compilar o modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Treinamento do modelo
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Avaliação final
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)

# Previsões e Relatório
y_pred = np.argmax(model.predict(x_test), axis=1)
y_true = y_test.flatten()
report = classification_report(y_true, y_pred, target_names=[f'Class {i}' for i in range(10)], output_dict=True)

# Matriz de Confusão
conf_matrix = confusion_matrix(y_true, y_pred)

# Gráficos e Relatório PDF
plt.figure(figsize=(10, 8))

# Gráfico de Acurácia e Perda
plt.subplot(2, 1, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title('Model Accuracy')

plt.subplot(2, 1, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.title('Model Loss')

# Matriz de Confusão
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=[f'Class {i}' for i in range(10)], yticklabels=[f'Class {i}' for i in range(10)])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')


with PdfPages('cifar10_training_report.pdf') as pdf:
    pdf.savefig(plt.figure(1))
    pdf.savefig(plt.figure(2))

print("Relatório completo foi salvo em 'cifar10_training_report.pdf'")
