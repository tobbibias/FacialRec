import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split


def make_pairs(images, labels):
    # making two lists , one for the images and one for the labels, we need these to make both pos and neg pairs.
    pairImages = []
    pairLabels = []
    # finding the total amount of "classes" people in the dataset.

    # and then build a list of indexes for each class label that
    # provides the indexes for all examples with a given label
    numClasses = len(np.unique(labels))
    idx = []
    labelIdx = []
    for l in np.unique(labels):
        indexes = []
        for i in range(0, len(images)):
            if l == labels[i]:
                indexes.append(i)
        idx.append(indexes)
        labelIdx.append(l)

    # loop over all images
    for idxA in range(len(images)):
        # grab the current image and label belonging to the current
        # iteration
        currentImage = images[idxA]
        label = labels[idxA]
        # randomly pick an image that belongs to the *same* class
        # label
        for i in range(0,len(labelIdx)):
            if labelIdx[i] == label:
                idxB = np.random.choice(idx[i])
        posImage = images[idxB]
        # prepare a positive pair and update the images and labels
        # lists, respectively
        pairImages.append([currentImage, posImage])
        pairLabels.append([1])
        # grab the indices for each of the class labels *not* equal to
        # the current label and randomly pick an image corresponding
        # to a label *not* equal to the current label
        negIdx = []
        for i in range(0,len(idx)):
            if label != labelIdx[i]:
                negIdx.append(idx[i])

        random = np.random.choice(np.array(negIdx,dtype='object').flatten())
        negImage = images[random[0]]
        # prepare a negative pair of images and update our lists
        pairImages.append([currentImage, negImage])
        pairLabels.append([0])

    # return a 2-tuple of our image pairs and labels
    return np.array(pairImages), np.array(pairLabels)


def loadData(path):
    # this function iterates over lfw and returns two lists
    # #one of images in the dataset and one containing the labels/names of the people depicted.
    img = []
    lables = []
    # iterates over all the people in the dataset
    for person in os.listdir(path):
        # for each person we add the picture and corresponding label to the output lists
        if person.startswith('.'):
            pass
        else:
            for picture in os.listdir(f'{path}/{person}'):
                if picture.startswith('.'):
                    pass
                else:
                    picture = cv2.imread(f'small_lfw/{person}/{picture}',1)
                    picture = cv2.resize(picture, (125, 125))
                    img.append(picture)
                    lables.append(person)
    trainX, testX, trainY, testY = train_test_split(img, lables, test_size=0.4)
    return trainX, testX, trainY, testY


def main():
    print('[Info] Loading data...')
    trainX, testX, trainY, testY = loadData('small_lfw')
    print('[Info] Making the training pairs...')
    (trainPair, trainLabel) = make_pairs(trainX,trainY)
    print('[Info] Making the testing pairs...')
    (testPair, testLabel) = make_pairs(testX,testY)
    print(f'[Info] Dataset fabrication complete.')


if __name__ == '__main__':
    main()
