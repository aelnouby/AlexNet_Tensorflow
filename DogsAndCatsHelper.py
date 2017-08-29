from glob import glob
from scipy import misc
import numpy as np


class DogsAndCatsHelper:

    @staticmethod
    def get_data():
        dir = 'add the dataset path here'
        train_dir = dir + 'train/'
        valid_dir = dir + 'valid/'

        train_images = []
        train_labels = []

        valid_images = []
        valid_labels = []

        # Dogs train
        for file in list(glob(train_dir + 'dogs/*.jpg')):
            img = misc.imread(file)
            img = misc.imresize(img, (227, 227))
            train_images.append(img)
            train_labels.append([1, 0])

        # Dogs valid
        for file in list(glob(valid_dir + 'dogs/*.jpg')):
            img = misc.imread(file)
            img = misc.imresize(img, (227, 227))
            valid_images.append(img)
            valid_labels.append([1, 0])

        # Cats train
        for file in list(glob(train_dir + 'cats/*.jpg')):
            img = misc.imread(file)
            img = misc.imresize(img, (227, 227))
            train_images.append(np.array(img))
            train_labels.append([0, 1])

        # Cats valid
        for file in list(glob(valid_dir + 'cats/*.jpg')):
            img = misc.imread(file)
            img = misc.imresize(img, (227, 227))
            valid_images.append(img)
            valid_labels.append([0, 1])

        perm = np.random.permutation(len(train_images))
        train_images = np.array(train_images)[perm]
        train_labels = np.array(train_labels)[perm]
        print(train_images.shape, train_labels.shape)
        valid_images = np.array(valid_images)
        valid_labels = np.array(valid_labels)

        return train_images, train_labels, valid_images, valid_labels
