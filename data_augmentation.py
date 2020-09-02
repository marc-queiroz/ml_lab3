import os

import numpy as np
import matplotlib.pyplot as plt


from time import time
import glob
from os import fdopen, remove
from tempfile import mkstemp
from shutil import move, copymode


from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator

from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates


class DataAugmentation():

    def __init__(self, train_file):
        self.train_paths = self.load_train_file(train_file)
        self.drive_path = './'
        self.aug_path = self.drive_path + "data/"
        self.train_file_aug = self.drive_path + 'train-aug.txt'
        # Train and Test files
        self.train_file = self.drive_path + 'train.txt'

    def load_train_file(self, train_file):
        arq = open(train_file, 'r')
        texto = arq.read()
        train_paths = texto.split('\n')
        # remove empty lines
        train_paths.remove('')
        train_paths.sort()
        return train_paths

    def replace(self, file_path, pattern):
        # Create temp file
        fh, abs_path = mkstemp()
        with fdopen(fh, 'w') as new_file:
            with open(file_path) as old_file:
                for line in old_file:
                    if pattern in line:
                        continue
                    new_file.write(line)
        # Copy the file permissions from the old file to the new file
        copymode(file_path, abs_path)
        # Remove original file
        remove(file_path)
        # Move new file
        move(abs_path, file_path)

    def save_to_aug(self, label, subdir):
        list_of_files = glob.glob(self.drive_path + self.aug_path + subdir + "/*")  # * means all if need specific format then *.csv
        if len(list_of_files) > 0:
            latest_file = max(list_of_files, key=os.path.getctime)
            self.replace(self.train_file_aug, latest_file.replace(self.aug_path + "/", ""))
            arq_aug = open(self.train_file_aug, "a+")
            arq_aug.write(latest_file.replace(self.aug_path + "/", "") + " " + label + "\n")
            arq_aug.close()

    def flip_rotation_brightness_zoom(self, path, zoom=[0.5, 1.0], brightness=[0.2, 1.0], rotation=90, flip_horizontal=False,
                                      flip_vertical=False, subdir="zoom"):
        path, label = path.split(' ')
        path = self.drive_path + self.aug_path + path
        img = load_img(path)
        # convert to numpy array
        data = img_to_array(img)
        # expand dimension to one sample
        samples = np.expand_dims(data, 0)
        # create image data augmentation generator
        datagen = ImageDataGenerator(zoom_range=zoom, brightness_range=brightness, rotation_range=rotation,
                                     horizontal_flip=flip_horizontal, vertical_flip=flip_vertical)
        # prepare iterator
        it = datagen.flow(samples, save_to_dir=self.aug_path + "/" + subdir + "/", batch_size=1)
        self.save_to_aug(label, subdir)
        # generate samples and plot
        for i in range(1):
            # define subplot
            plt.subplot(330 + 1 + i)
            # generate batch of images
            batch = it.next()
            # convert to unsigned integers for viewing
            image = batch[0].astype('uint8')
            # plot raw pixel data
            plt.imshow(image)
        # show the figure
        # plt.show()

    def random_zoom(self, path, zoom=[0.5, 1.0], subdir="zoom"):
        path, label = path.split(' ')
        path = self.drive_path + self.aug_path + path
        img = load_img(path)
        # convert to numpy array
        data = img_to_array(img)
        # expand dimension to one sample
        samples = np.expand_dims(data, 0)
        # create image data augmentation generator
        datagen = ImageDataGenerator(zoom_range=zoom)
        # prepare iterator
        it = datagen.flow(samples, save_to_dir=self.aug_path + "/" + subdir + "/", batch_size=1)
        self.save_to_aug(label, subdir)
        # generate samples and plot
        for i in range(1):
            # define subplot
            plt.subplot(330 + 1 + i)
            # generate batch of images
            batch = it.next()
            # convert to unsigned integers for viewing
            image = batch[0].astype('uint8')
            # plot raw pixel data
            plt.imshow(image)
        # show the figure
        # plt.show()

    def random_brightness(self, path, brightness=[0.2, 1.0], subdir="brightness"):
        path, label = path.split(' ')
        path = self.drive_path + self.aug_path + path
        # load the image
        img = load_img(path)
        # convert to numpy array
        data = img_to_array(img)
        # expand dimension to one sample
        samples = np.expand_dims(data, 0)
        # create image data augmentation generator
        datagen = ImageDataGenerator(brightness_range=brightness)
        # prepare iterator
        it = datagen.flow(samples, save_to_dir=self.aug_path + "/" + subdir + "/", batch_size=1)
        self.save_to_aug(label, subdir)
        # generate samples and plot
        for i in range(1):
            # define subplot
            plt.subplot(330 + 1 + i)
            # generate batch of images
            batch = it.next()
            # convert to unsigned integers for viewing
            image = batch[0].astype('uint8')
            # plot raw pixel data
            plt.imshow(image)
        # show the figure
        # plt.show()

    def random_rotation(self, path, rotation=90, subdir="rotation"):
        path, label = path.split(' ')
        path = self.drive_path + self.aug_path + path
        img = load_img(path)
        # convert to numpy array
        data = img_to_array(img)
        # expand dimension to one sample
        samples = np.expand_dims(data, 0)
        # create image data augmentation generator
        datagen = ImageDataGenerator(rotation_range=rotation)
        # prepare iterator
        it = datagen.flow(samples, save_to_dir=self.aug_path + "/" + subdir + "/", batch_size=1)
        self.save_to_aug(label, subdir)
        # generate samples and plot
        for i in range(1):
            # define subplot
            plt.subplot(330 + 1 + i)
            # generate batch of images
            batch = it.next()
            # convert to unsigned integers for viewing
            image = batch[0].astype('uint8')
            # plot raw pixel data
            plt.imshow(image)
        # show the figure
        # plt.show()

    def horizontal_vertical_flip(self, path, flip_horizontal=False, flip_vertical=False, subdir="flip"):
        path, label = path.split(' ')
        path = self.drive_path + self.aug_path + path
        # load the image
        img = load_img(path)
        # convert to numpy array
        data = img_to_array(img)
        # expand dimension to one sample
        samples = np.expand_dims(data, 0)
        # create image data augmentation generator
        datagen = ImageDataGenerator(horizontal_flip=flip_horizontal, vertical_flip=flip_vertical)
        # prepare iterator
        it = datagen.flow(samples, save_to_dir=self.aug_path + "/" + subdir + "/", batch_size=1)
        self.save_to_aug(label, subdir)
        # generate samples and plot
        for i in range(1):
            # define subplot
            plt.subplot(330 + 1 + i)
            # generate batch of images
            batch = it.next()
            # convert to unsigned integers for viewing
            image = batch[0].astype('uint8')
            # plot raw pixel data
            plt.imshow(image)
        # show the figure
        plt.show()

    def horizontal_vertical_shift(self, path, size=0.5, bool_width=True, subdir="shift"):
        path, label = path.split(' ')
        path = self.drive_path + self.aug_path + path
        # load the image
        print('path', path)
        img = load_img(path)
        # convert to numpy array
        data = img_to_array(img)
        # expand dimension to one sample
        samples = np.expand_dims(data, 0)
        # create image data augmentation generator
        if bool_width:
            datagen = ImageDataGenerator(width_shift_range=[-size, size], fill_mode='wrap')
        else:
            datagen = ImageDataGenerator(height_shift_range=size)
        # prepare iterator
        it = datagen.flow(samples, save_to_dir=self.aug_path + "/" + subdir + "/", batch_size=1)
        self.save_to_aug(label, subdir)
        # generate samples and plot
        for i in range(1):
            # define subplot
            plt.subplot(330 + 1 + i)
            # generate batch of images
            batch = it.next()
            # convert to unsigned integers for viewing
            image = batch[0].astype('uint8')
            # plot raw pixel data
            plt.imshow(image)
        # show the figure
        plt.show()

    def elastic_transform(self, image, alpha_range, sigma, random_state=None):
        """Elastic deformation of images as described in [Simard2003]_.
        .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
           Convolutional Neural Networks applied to Visual Document Analysis", in
           Proc. of the International Conference on Document Analysis and
           Recognition, 2003.
       # Arguments
           image: Numpy array with shape (height, width, channels). 
           alpha_range: Float for fixed value or [lower, upper] for random value from uniform distribution.
               Controls intensity of deformation.
           sigma: Float, sigma of gaussian filter that smooths the displacement fields.
           random_state: `numpy.random.RandomState` object for generating displacement fields.
        """

        if random_state is None:
            random_state = np.random.RandomState(None)

        if np.isscalar(alpha_range):
            alpha = alpha_range
        else:
            alpha = np.random.uniform(low=alpha_range[0], high=alpha_range[1])

        shape = image.shape
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

        x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
        indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1)), np.reshape(z, (-1, 1))

        return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)

    def elastic_transformation(self, path, width_shift_range=0, height_shift_range=0, bool_width=True, subdir="elastic"):
        path, label = path.split(' ')
        path = self.drive_path + self.aug_path + path
        # load the image
        print('path', path)
        img = load_img(path)
        # convert to numpy array
        data = img_to_array(img)
        # expand dimension to one sample
        samples = np.expand_dims(data, 0)
        # create image data augmentation generator
        datagen = ImageDataGenerator(width_shift_range, height_shift_range, preprocessing_function=lambda x: self.elastic_transform(x, alpha_range=[8, 10], sigma=3))
        # prepare iterator
        it = datagen.flow(samples, save_to_dir=self.aug_path + "/" + subdir + "/", batch_size=1)
        self.save_to_aug(label, subdir)
        # generate samples and plot
        for i in range(1):
            # define subplot
            plt.subplot(330 + 1 + i)
            # generate batch of images
            batch = it.next()
            # convert to unsigned integers for viewing
            image = batch[0].astype('uint8')
            # plot raw pixel data
            plt.imshow(image)
        # show the figure
        # plt.show()

    def generate_images(self):
        for image_path in self.train_paths:
            print('IMAGE PATH', image_path)
            # self.horizontal_vertical_flip(image_path, flip_horizontal=True, flip_vertical=False)
            # self.horizontal_vertical_flip(image_path, flip_horizontal=False, flip_vertical=True)
            # self.horizontal_vertical_flip(image_path, flip_horizontal=True, flip_vertical=True)
            # self.horizontal_vertical_flip(image_path, flip_horizontal=False, flip_vertical=False)
            self.elastic_transformation(image_path)
            # self.horizontal_vertical_shift(image_path, bool_width=True)
            # self.horizontal_vertical_shift(image_path, bool_width=False)
            # self.random_rotation(image_path, rotation=10)
            # self.random_rotation(image_path, rotation=20)
            # self.random_rotation(image_path, rotation=30)
            # self.random_rotation(image_path, rotation=45)
            # self.random_brightness(image_path)
            # self.random_brightness(image_path, brightness=[0.1, 0.2])
            # self.random_brightness(image_path, brightness=[0.3, 0.4])
            # self.random_brightness(image_path, brightness=[0.4, 0.5])
            # self.random_zoom(image_path)
            # self.random_zoom(image_path, zoom=[0.1, 0.5])
            # self.random_zoom(image_path, zoom=[0.1, 0.2])
            # self.random_zoom(image_path, zoom=[0.1, 0.3])
            # self.flip_rotation_brightness_zoom(image_path, rotation=30)
            # self.flip_rotation_brightness_zoom(image_path, zoom=[0.1, 0.5], brightness=[0.1, 0.5])
            # self.flip_rotation_brightness_zoom(image_path, zoom=[0.1, 0.5], brightness=[0.1, 0.5])
            # self.flip_rotation_brightness_zoom(image_path, zoom=[0.1, 0.5], brightness=[0.1, 0.5], rotation=30)
            # self.flip_rotation_brightness_zoom(image_path, zoom=[0.1, 0.5], brightness=[0.1, 0.5], rotation=30)
            # self.flip_rotation_brightness_zoom(image_path, zoom=[0.1, 0.5], brightness=[0.1, 0.5], rotation=30)
            # self.flip_rotation_brightness_zoom(image_path, zoom=[0.1, 0.8], brightness=[0.1, 0.8], rotation=45)
            # self.flip_rotation_brightness_zoom(image_path, zoom=[0.1, 0.8], brightness=[0.1, 0.8], rotation=45)
            # self.flip_rotation_brightness_zoom(image_path, zoom=[0.1, 0.2], brightness=[0.1, 0.2], rotation=30)
            # self.flip_rotation_brightness_zoom(image_path, zoom=[0.1, 0.2], brightness=[0.1, 0.2], rotation=45)
            # self.flip_rotation_brightness_zoom(image_path, zoom=[0.9, 1], brightness=[0.9, 1], rotation=30)
            # self.flip_rotation_brightness_zoom(image_path, zoom=[0.9, 1], brightness=[0.9, 1], rotation=45)


if __name__ == '__main__':
    # Execution Time
    start = time()
    dg = DataAugmentation('train.txt')
    dg.generate_images()
    print(f'Execution Time: {time() - start}')
    print('GERANDO AS NOVAS IMAGENS - DONE')
