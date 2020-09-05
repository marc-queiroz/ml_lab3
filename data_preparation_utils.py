import cv2
import time
import numpy as np

import matplotlib.pyplot as plt

from PIL import Image

import keras

from sklearn.metrics import confusion_matrix

from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input

from keras.preprocessing import image


def test():
    return np.zeros((10, 10))


def resize_data(data, size, convert):
    if convert:
        data_upscaled = np.zeros((data.shape[0], size[0], size[1], 3))
    else:
        data_upscaled = np.zeros((data.shape[0], size[0], size[1]))
        for i, img in enumerate(data):
            large_img = cv2.resize(img, dsize=(
                size[1], size[0]), interpolation=cv2.INTER_CUBIC)
            data_upscaled[i] = large_img
        return data_upscaled


def load_images(image_paths, convert=False):
    x = []
    y = []
    for image_path in image_paths:
        path, label = image_path.split(' ')
        path = './ml_lab3/' + path
        if convert:
            image_pil = Image.open(path).convert('RGB')
        else:
            image_pil = Image.open(path).convert('L')
        img = np.array(image_pil, dtype=np.uint8)
        x.append(img)
        y.append([int(label)])
    x = np.array(x)
    y = np.array(y)
    if np.min(y) != 0:
        y = y-1
    return x, y


def load_dataset(train_file, test_file, resize, convert=False, size=(224, 224)):
    arq = open(train_file, 'r')
    texto = arq.read()
    train_paths = texto.split('\n')
    print('Size:', size)
    train_paths.remove('')
    train_paths.sort()
    print("Loading training set...")
    x_train, y_train = load_images(train_paths, convert)
    arq = open(test_file, 'r')
    texto = arq.read()
    test_paths = texto.split('\n')
    test_paths.remove('')
    test_paths.sort()
    print("Loading testing set...")
    x_test, y_test = load_images(test_paths, convert)
    if resize:
        print("Resizing images...")
        x_train = resize_data(x_train, size, convert)
        x_test = resize_data(x_test, size, convert)
    if not convert:
        x_train = x_train.reshape(x_train.shape[0], size[0], size[1], 1)
        x_test = x_test.reshape(x_test.shape[0], size[0], size[1], 1)
    print(np.shape(x_train))
    return (x_train, y_train), (x_test, y_test)


def generate_labels(x_test, y_test):
    labels = []
    for i in range(len(x_test)):
        labels.append(y_test[i][0])
    return labels


def normalize_images(x):
    x = x.astype('float32')
    x /= 255
    return x


def convert_vector(x, num_classes):
    return keras.utils.to_categorical(x, num_classes)


def fit_model(model, x_train, y_train, x_test, y_test, epochs, batch_size=128, verbose=1):
    return model.fit(x=x_train, y=y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test), verbose=verbose)


def get_confusion_matrix(model, x_test, labels):
    pred = []
    y_pred = model.predict_classes(x_test)
    for i in range(len(x_test)):
        pred.append(y_pred[i])
    return confusion_matrix(labels, pred)


def plot_graphs(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))
    plt.figure(figsize=(12, 9))
    plt.plot(epochs, acc, 'b', label='Acurácia do treinamento')
    plt.plot(epochs, val_acc, 'r', label='Acurácia da validação')
    plt.title('Acurácia do treinamento e validação')
    plt.legend()
    plt.figure(figsize=(12, 9))
    plt.plot(epochs, loss, 'b', label='Perda do treinamento')
    plt.plot(epochs, val_loss, 'r', label='Perda da validação')
    plt.title('Perda do treinamento e validação')
    plt.legend()
    plt.show()


def extract_features(input_file, output_file, img_rows, img_cols, dir_dataset):
    file_input = open(input_file, 'r')
    input = file_input.readlines()
    file_input.close()
    output = open(output_file, 'w')
    model = InceptionV3(weights='imagenet', include_top=False)
    for i in input:
        sample_name, sample_class = i.split()
        img_path = dir_dataset + sample_name
        print(img_path)
        img = image.load_img(img_path, target_size=(img_rows, img_cols))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        inception_features = model.predict(img_data)
        features_np = np.array(inception_features)
        features_np = features_np.flatten()
        output.write(sample_class+' ')
        for j in range(features_np.size):
            output.write(str(j+1)+':'+str(features_np[j])+' ')
        output.write('\n')
    print(features_np.size)
    output.close()


def round_float(value):
    return float("{:.3f}".format(value))


def get_time():
    return time.time()


def get_time_diff(start_time):
    end_time = time.time()
    return round_float(end_time - start_time)


def plot_confusion_matrix(cm):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    for (x, y), value in np.ndenumerate(cm):
        plt.text(x, y, f"{value:.0f}", va="center", ha="center")
    plt.title('Matrix de Confusão')
    fig.colorbar(cax)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
