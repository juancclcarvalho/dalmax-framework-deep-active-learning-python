# Usage: python test.py --dir_test DATA/DATA_CIFAR10/test/ --dir_results results/ --model results/active_learning_model.h5

# System imports
import os
import sys
import argparse
import time

# TensorFlow and Sklearn
import tensorflow as tf # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from sklearn.metrics import accuracy_score # type: ignore

# Add path to root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Local import
from utils.utilities import load_images

def valid_args(args):
    modal_path, dir_test, dir_results = args.model, args.dir_test, args.dir_results
    if not os.path.exists(modal_path):
        raise FileNotFoundError(f"Model not found: {modal_path}")
    if not os.path.exists(dir_test):
        raise FileNotFoundError(f"Test dataset not found: {dir_test}")
    if not os.path.exists(dir_results):
        raise FileNotFoundError(f"Results directory not found: {dir_results}")
    ######################
def predict(target_classes_names, image, reloaded):
    import numpy as np
    ti = time.time()
    probabilities = reloaded.predict(np.asarray([image]))[0]
    t = time.time() - ti
    class_idx = np.argmax(probabilities)
    confidence = probabilities[class_idx]
    class_label = None
    # For no dicionario target_classes_names
    for key, value in target_classes_names.items():
        # If value is equal to class_idx: achou a classe
        if value == class_idx:
            class_label = key
    
    return class_label, confidence,  t

def load_image(filename, img_size):
    img_path = filename
    from tensorflow.keras.preprocessing import image # type: ignore

    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0

    return img_array


def main(args):
    # Setup paths
    modal_path = args.model
    dir_test = args.dir_test
    dir_results = args.dir_results
    img_size = args.img_size
    # Create if not exists
    if not os.path.exists(dir_results):
        os.makedirs(dir_results)

    # Load the model
    model = load_model(modal_path)

    testing_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    
    testing_generator = testing_datagen.flow_from_directory(
        dir_test, 
        shuffle=False, 
        seed=42,
        color_mode="rgb", 
        class_mode="categorical",
        target_size=(img_size, img_size),
        batch_size=64)
    
    # Record the time
    init_time = time.time()


    print("\n\n\n\n-------------------------------------")

    
    # Avaliação final
    test_images, test_labels, label_map, __ = load_images(dir_test, (128, 128))
    print(f"Test images shape: {test_images.shape}")
    print(f"Test labels shape: {test_labels.shape}")
    print(f"Label map: {label_map}")
    test_images = test_images / 255.0
    test_labels = to_categorical(test_labels, num_classes=len(label_map))
    predictions = model.predict(test_images, batch_size=1, verbose=1).argmax(axis=1)
    
    end_time = time.time()
    text_time = f"Total time: {end_time - init_time:.2f} seconds"
    
    accuracy = accuracy_score(test_labels.argmax(axis=1), predictions)
    text_final = f"Final Test Accuracy: {accuracy * 100:.2f}%"
   

    # Save on file the final accuracy
    with open(f'{dir_results}/final_accuracy.txt', 'w') as f:
        f.write(text_final)
    

    # Save time on infos file
    with open(f'{dir_results}/infos.txt', 'w') as f:
        f.write(text_time)
    
    print(text_time)
    print(text_final)

    print("\n\n\n\n-------------------------------------")

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    result_evaluate = model.evaluate(testing_generator, verbose=1,  batch_size=1)

    print(f'Final loss evaluate: {result_evaluate[0]}')
    print(f'Final accuracy evaluate: {result_evaluate[1]}')
    print(f"evaluate.metrics_names: {model.metrics_names}")

    print(f"Results saved on {dir_results}")
    print("Done!")


    # Set path the image
    img_path = 'DATA/daninhas_balanceado/train/DATASET_BRACHIARIA/ORTOFOTO_1_8411_18677_0.jpg'

    # Load image
    img = load_image(img_path, (img_size, img_size))
    # print(img)

    #Predict image
    class_label, confidence, time_predited = predict(label_map, img, model)

    title_text  = "\n\n\n\n\nPredicted class: %s \nConfidence: %f" % (class_label, confidence)
    title_text += "\nTime predict: " + str(time_predited) 
    subtitle_text = "Time predict: " + str(time_predited) 

    # print(title_text)
    import matplotlib.pyplot as plt 
    plt.figure(figsize=(10,10))
    plt.subplots_adjust(hspace=0.2)

    plt.imshow(img)
    plt.title(subtitle_text)
    plt.axis('off')
    _ = plt.suptitle(title_text)

    plt.savefig(dir_results + "/result_test_one_image_tf_model_testing_predict.pdf")


    # Chamar a função load_image e predict para todas as imagens do diretório de teste. E calcular a acurácia para cada imagem e fazer a acurácia média de todas as imagens.
    classes_diretorios_lista = os.listdir(dir_test)
    accuracy_images = []
    for classe_real in classes_diretorios_lista:
        dir_classe = os.path.join(dir_test, classe_real)
        images_classe = os.listdir(dir_classe)
        for image_name in images_classe:
            img_path = os.path.join(dir_classe, image_name)
            img = load_image(img_path, (img_size, img_size))
            class_label, confidence, time_predited = predict(label_map, img, model)
            if class_label == classe_real:
                accuracy_images.append(1)
            else:
                accuracy_images.append(0)

    accuracy_images = sum(accuracy_images) / len(accuracy_images)
    print(f"Accuracy images: {accuracy_images}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Active Learning Model')
    parser.add_argument('--dir_test', type=str, default='DATA/DATA_CIFAR10/test/', help='Test dataset directory')
    parser.add_argument('--dir_results', type=str, default='results/', help='Results directory')
    parser.add_argument('--model', type=str, default='results/active_learning_model.h5', help='Model path')
    parser.add_argument('--img_size', type=int, default=128, help='Image size')

    args = parser.parse_args()

    valid_args(args)
    main(args)