import os
import datetime
IMG_SHAPE = (125, 125, 3)
BATCH_SIZE = 16
EPOCHS = 25
BASE_OUTPUT = 'output'
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, 'res_siamese_model'])
# MODEL_PATH = os.path.sep.join([BASE_OUTPUT, 'siamese_model'])
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, 'PLOT.png'])
LOG_PATH = 'output/logs/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
