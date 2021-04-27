from facialrec.build_resnet import resnet50
from facialrec import utils
from facialrec import config
import build_img_pairs
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda

# loading data
print('[Data] Loading the dataset.')
trainX, testX, trainY, testY = build_img_pairs.loadData('small_lfw')
print('[Data] Making the training samples.')
trainPair, trainLabel = build_img_pairs.make_pairs(trainX,trainY)
print('[Data] Done!')
print('[Data] Making the testing samples.')
testPair, testLabel = build_img_pairs.make_pairs(testX,testY)
trainPair = trainPair[:280]
trainLabel = trainLabel[:280]
testPair = testPair[:120]
testLabel = testLabel[:120]

print('[Data] Done!')


# building the network
print('[Build] Building network architecture.')
imgA = Input(shape=config.IMG_SHAPE)
imgB = Input(shape=config.IMG_SHAPE)

featureExtractor = resnet50(config.IMG_SHAPE)
featsA = featureExtractor(imgA)
featsB = featureExtractor(imgB)
distance = Lambda(utils.euclidean_distance)([featsA, featsB])
outputs = Dense(1, activation="sigmoid")(distance)
model = Model(inputs=[imgA, imgB], outputs=outputs)

# compile the model
print("[INFO] compiling model...")
opt = tf.keras.optimizers.SGD(learning_rate=0.1)
model.compile(loss="binary_crossentropy", optimizer='Adam',
	metrics=["accuracy"])

# train the model
print("[INFO] training model...")
# setting up tensorboard
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=config.LOG_PATH, histogram_freq=1)

history = model.fit(
    [trainPair[:, 0], trainPair[:, 1]], trainLabel[:],
    validation_data=([testPair[:, 0], testPair[:, 1]], testLabel[:]),
    callbacks= [tensorboard_callback],
    batch_size=config.BATCH_SIZE,
    epochs=config.EPOCHS)
# serialize the model to disk
print("[INFO] saving siamese model...")
model.save(config.MODEL_PATH)
# plot the training history
print("[INFO] plotting training history...")
utils.plot_training(history, config.PLOT_PATH)