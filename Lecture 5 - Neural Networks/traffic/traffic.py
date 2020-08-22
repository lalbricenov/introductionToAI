import cv2
import numpy as np
import os
import sys
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])
    plt.figure()
    plt.grid(False)
    # for i in range(10):
    #     print(images[i].shape)
    #     plt.imshow(images[i])
    #     plt.colorbar()
    #     plt.show()
    # Split data into training and testing sets

    labels = tf.keras.utils.to_categorical(labels)
    # print(f'X: {np.array(images).shape}')
    # print(f'Y: {np.array(labels).shape}')

    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    images = []
    labels = []
    folderNames = os.listdir(data_dir)
    count = 0
    for folderName in folderNames:
        imgNames = os.listdir(os.path.join(data_dir, folderName))
        # print(imgNames)
        for imgName in imgNames:
            img = cv2.imread(os.path.join(data_dir, folderName, imgName))
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            # print()
            images.append(img)
            labels.append(int(folderName))
        count += 1
        print(f'Type {count}/{len(folderNames)} read')
    return (images, labels)


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """

    # Dense means that each of the nodes in the layer will be connected to every other node in the previous layer
    model = tf.keras.models.Sequential(
        [
            # Learn 32 different filters
            tf.keras.layers.Conv2D(
                32, (3, 3), input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
            # MaxPooling2D reduces the size of the images
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(NUM_CATEGORIES, activation="sigmoid")
        ]
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


if __name__ == "__main__":
    main()
