      #LIBRARIES AND IMPORT
#install libararies
#pip install tensorflow
#pip install livelossplot

# Import essential libraries for data manipulation, visualization, and deep learning
import os
import numpy as np
import pandas as pd
import random
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.image as mimg
%matplotlib inline
from PIL import Image
from scipy import misc
import tensorflow as tf

import keras
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model

from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, Concatenate, BatchNormalization
from keras.layers import UpSampling2D, Dropout, Add, Multiply, Subtract, AveragePooling2D
from keras.layers.core import Activation, SpatialDropout2D
from keras.layers.core import Dense, Lambda
from keras.layers import GlobalAveragePooling2D, Reshape, Dense, Permute, Flatten

from tensorflow.keras.utils import plot_model
from tensorflow.keras.applications import DenseNet121
from collections import Counter
from mlxtend.plotting import plot_confusion_matrix

from keras.optimizers import * 
from keras.callbacks import *
from keras.activations import *

from sklearn.metrics import classification_report, confusion_matrix

np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)
        #DATA LOADING 
train_loc = '/kaggle/input/mpox-skin-lesion-dataset-version-20-msld-v20/Augmented Images/Augmented Images/FOLDS_AUG/fold5_AUG/Train'
val_loc = '/kaggle/input/mpox-skin-lesion-dataset-version-20-msld-v20/Original Images/Original Images/FOLDS/fold5/Valid'
test_loc = '/kaggle/input/mpox-skin-lesion-dataset-version-20-msld-v20/Original Images/Original Images/FOLDS/fold5/Test'

BATCH_SIZE = 16

trdata = ImageDataGenerator()
train_data = trdata.flow_from_directory(directory=train_loc, target_size=(224,224),batch_size=BATCH_SIZE, shuffle=True,
                                               seed=42)

vdata = ImageDataGenerator()
val_data = vdata.flow_from_directory(directory=val_loc, target_size=(224,224),batch_size=BATCH_SIZE, shuffle=True,
                                               seed=42)

tsdata = ImageDataGenerator()
test_data = tsdata.flow_from_directory(directory=test_loc, target_size=(224,224),batch_size=BATCH_SIZE, shuffle=False, seed = 42)

train_data.class_indices
      #MODEL CREATION
def create_model(input_shape, n_classes , optimizer, fine_tune):
    """
    Compiles a model integrated with VGG16 pretrained layers
    
    input_shape: tuple - the shape of input images (width, height, channels)
    n_classes: int - number of classes for the output layer
    optimizer: string - instantiated optimizer to use for training. Defaults to 'RMSProp'
    fine_tune: int - The number of pre-trained layers to unfreeze.
                If set to 0, all pretrained layers will freeze during training
    """
    
    # Pretrained convolutional layers are loaded using the Imagenet weights.
    # Include_top is set to False, in order to exclude the model's fully-connected layers.
    conv_base = DenseNet121(include_top=False,
                     weights='imagenet', 
                     input_shape=input_shape)
    
    # Initialize the model with weights from HAM10000 keeping by name True
    #conv_base.load_weights("/kaggle/input/weightofham/DenseNet121.h5",by_name=True)
    
    # Defines how many layers to freeze during training.
    # Layers in the convolutional base are switched from trainable to non-trainable
    # depending on the size of the fine-tuning parameter.
#     if fine_tune > 0:
#         for layer in conv_base.layers[:-fine_tune]:
#             layer.trainable = False
#     else:
#         for layer in conv_base.layers:
#             layer.trainable = False

    # Create a new 'top' of the model (i.e. fully-connected layers).
    # This is 'bootstrapping' a new top_model onto the pretrained layers.
    top_model = conv_base.output
    top_model = Flatten(name="flatten")(top_model)
    top_model = Dense(1024, activation='relu')(top_model)
    top_model = Dense(512, activation='relu')(top_model)
    top_model = Dropout(0.2)(top_model)    
    top_model = Dense(256, activation='relu')(top_model)
    top_model = Dropout(0.15)(top_model)
    output_layer = Dense(n_classes, activation='softmax')(top_model)
    
    # Group the convolutional base and new fully-connected layers into a Model object.
    model = Model(inputs=conv_base.input, outputs=output_layer)

    # Compiles the model for training.
    model.compile(optimizer=opt, 
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    
    
    model.summary()
    
    return model
    #MODEL TRAINING AND SET UP
input_shape = (224, 224, 3)
opt = tf.keras.optimizers.Adam(learning_rate=0.00001)
n_classes = 6
ft=0

# First we'll train the model without Fine-tuning
model = create_model(input_shape, n_classes, opt, ft)
plot_model(
    model,
    to_file="model.png",
    show_shapes=True,
    #show_dtype=True,
    show_layer_names=True,
    rankdir="TB",
    expand_nested=False,
    dpi=96,
)
STEP_SIZE_TRAIN = train_data.n//train_data.batch_size
STEP_SIZE_VALID = val_data.n//test_data.batch_size
#n_epochs = 100

    #CLASS WEIGHTS
from collections import Counter
counter = Counter(train_data.classes)                       
max_val = float(max(counter.values()))   
class_weights = {class_id : max_val/num_images for class_id, num_images in counter.items()}
class_weights

      #MODEL TRAINING
from livelossplot import PlotLossesKeras

checkpoint = ModelCheckpoint("../working/DenseNet121Full.h5", monitor='val_accuracy', verbose=1, 
                             save_best_only=True, save_weights_only=True, mode='auto')
early_stop = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=20, verbose=1, mode='auto', restore_best_weights=True)

history = model.fit(train_data,
                    epochs =100,
                    steps_per_epoch = STEP_SIZE_TRAIN,
                    class_weight = class_weights,
                    validation_data = val_data,
                    validation_steps = STEP_SIZE_VALID,
                    callbacks=[checkpoint, early_stop, PlotLossesKeras()]
                    )
      #MODEL EVALUATION
model_preds = model.predict(test_data,test_data.samples//test_data.batch_size+1)
model_pred_classes = np.argmax(model_preds , axis=1)


from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
true_classes = test_data.classes
acc = accuracy_score(true_classes, model_pred_classes)
print("DenseNet121-based Model Accuracy: {:.2f}%".format(acc * 100))

print('Precision: %.3f' % precision_score(true_classes, model_pred_classes,average='macro'))
print('Recall: %.3f' % recall_score(true_classes, model_pred_classes,average='macro'))
print('F1 Score: %.3f' % f1_score(true_classes, model_pred_classes,average='macro'))


x = confusion_matrix(test_data.classes, model_pred_classes)

plot_confusion_matrix(x)
print('Classification Report')
target_names = ['Chickenpox','Cowpox','HFMD','Healthy','Measles','Monkeypox']
print(classification_report(test_data.classes, model_pred_classes))

# Get the names of the ten classes
class_names = test_data.class_indices.keys()

def plot_heatmap(y_true, y_pred, class_names, ax, title):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm, 
        annot=True, 
        square=True, 
        xticklabels=class_names, 
        yticklabels=class_names,
        fmt='d', 
        cmap=plt.cm.Greens, #Blues, YlGnBu, YlOrRd
        cbar=False,
        ax=ax
    )
    ax.set_title(title, fontsize=16)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)

#fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))
fig, (ax1) = plt.subplots(1, 1, figsize=(20, 10))
plot_heatmap(true_classes, model_pred_classes, class_names, ax1, title="DenseNet121")    
# plot_heatmap(true_classes, vgg_pred_classes, class_names, ax2, title="Transfer Learning (VGG16) No Fine-Tuning")    
# plot_heatmap(true_classes, vgg_pred_classes_ft, class_names, ax3, title="Transfer Learning (VGG16) with Fine-Tuning")    

# fig.suptitle("Confusion Matrix Model Comparison", fontsize=24)
# fig.tight_layout()
# fig.subplots_adjust(top=1.25)
# plt.show()
        #SAVE MODEL 
#Save the intire model
model.save('SkinNet.keras')
