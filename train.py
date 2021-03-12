from facialrec.siamese_net import build_model
from facialrec import utils
from facialrec import config
import build_img_pairs
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda


print('[Data] Loading the dataset.')
trainX, testX, trainY, testY = build_img_pairs.loadData('lfw')
print('[Data] Making the training samples.')
trainPair,trainLabel = build_img_pairs.make_pairs(trainX,trainY)
print('[Data] Done!')
print('[Data] Making the testing samples.')
testPair, testLabel = build_img_pairs.make_pairs(testX,testY)
print('[Data] Done!')
print('[Build] Building network architecture.')
imgA = Input(shape=config.IMG_SHAPE)
imgB = Input(shape=config.IMG_SHAPE)
featureExtractor = build_model(config.IMG_SHAPE)
featsA = featureExtractor(imgA)
featsB = featureExtractor(imgB)
# finally, construct the siamese network
distance = Lambda(utils.euclidean_distance)([featsA, featsB])
outputs = Dense(1, activation="sigmoid")(distance)
model = Model(inputs=[imgA, imgB], outputs=outputs)

# compile the model
print("[INFO] compiling model...")
model.compile(loss="binary_crossentropy", optimizer="adam",
	metrics=["accuracy"])
# train the model
print("[INFO] training model...")
history = model.fit(
    [trainPair[:, 0], trainPair[:, 1]], trainLabel[:],
    validation_data=([testPair[:, 0], testPair[:, 1]], testLabel[:]),
    batch_size=config.BATCH_SIZE,
    epochs=config.EPOCHS)
# serialize the model to disk
print("[INFO] saving siamese model...")
model.save(config.MODEL_PATH)
# plot the training history
print("[INFO] plotting training history...")
utils.plot_training(history, config.PLOT_PATH)