import csv
import scipy.misc as spm
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D


def train_model():
    the_path = '../training images/'

    images = []
    angles = []
    throw_aways = 0
    with open(the_path + 'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:

            center_path = row[0]
            left_path = row[1]
            right_path = row[2]

            # Try loading the center image, if it doesn't exist skip this row.
            # This allows for easily throwing away some bad sets of training data.
            try:
                center_image = spm.imread(center_path)
            except:
                print('error: image {} not found'.format(center_path))
                continue

            center_angle = float(row[3])

            # Throw away a percentage of small angle data so most of the training
            # is concentrated on turning away from the edge of the track.
            if abs(center_angle) < 0.2:
                chance = random.randint(1, 100)
                if chance <= 80:
                    throw_aways += 1
                    continue

            images.append(center_image)
            angles.append(center_angle)

            correction = 0.2
            angle_left = center_angle + correction
            angle_right = center_angle - correction

            angles.append(angle_left)
            images.append(spm.imread(left_path))

            angles.append(angle_right)
            images.append(spm.imread(right_path))

            # Also append flipped center images and measurements.
            images.append(np.fliplr(center_image))
            angles.append(-center_angle)

    print('threw away {} small angle samples'.format(throw_aways))

    X_train = np.array(images)
    y_train = np.array(angles)

    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((60, 10), (0, 0))))

    model.add(Conv2D(6, 5, activation='relu'))
    model.add(MaxPool2D())
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(20))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, epochs=4, validation_split=0.2, shuffle=True, verbose=1)

    model.save('model.h5')


if __name__ == '__main__':
    train_model()
