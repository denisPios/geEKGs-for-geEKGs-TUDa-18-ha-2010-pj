import tensorflow as tf
from tensorflow import keras
from keras.models import model_from_json
from tensorflow.keras.layers import Flatten, Dense, Conv1D, MaxPool1D, Dropout
from keras.layers import GaussianNoise
import os
from process_data import  generate_data_set,filter_ecg
import numpy as np

#this trains 
def train(ecg_leads,ecg_labels,ecg_names,fs,num_HB,model_name:str='Abgabe',epochs:int=8,batch_size:int=64,weights={0: 2.,1: 10.,2: 1.,3: 1}):
    #filter data
    for idx, ecg_lead in enumerate(ecg_leads):
        ecg_leads[idx]=filter_ecg(ecg_leads[idx])
        ecg_leads[idx]=ecg_leads[idx]/np.amax(ecg_leads[idx])
    #rename labels to int
    for idx, label in enumerate(ecg_labels):
        if label=="N":
            ecg_labels[idx]=0
        if label=="A":
            ecg_labels[idx]=1
        if label=="O":
            ecg_labels[idx]=2
        if label=="~":
            ecg_labels[idx]=3

    train=generate_data_set(num_HB,ecg_leads,ecg_names,fs,ecg_labels,smote=True)
    #train is a dictionary, get the actual variables
    x_train=train['input']
    y_train=train['labels']
    opt_length=train['optimal_length']
    
        # Create sequential model 
    cnn_model = tf.keras.models.Sequential()
    #First CNN layer  with 32 filters, kernel 5
    cnn_model.add(Conv1D(filters=128, kernel_size=(50,), strides=3, padding='same',
                         activation=tf.keras.layers.LeakyReLU(alpha=0.001), input_shape = (x_train.shape[1:])))
    cnn_model.add(tf.keras.layers.BatchNormalization())
    cnn_model.add(MaxPool1D(pool_size=(2,), strides=3, padding='same'))
    #Second CNN layer same as first
    cnn_model.add(Conv1D(filters=32, kernel_size=(7,), strides=1, padding='same',
                         activation=tf.keras.layers.LeakyReLU(alpha=0.001)))
    cnn_model.add(tf.keras.layers.BatchNormalization())
    cnn_model.add(MaxPool1D(pool_size=(2,), strides=2, padding='same'))
    cnn_model.add(Dropout(0.5))
    #Third CCN layer with 64 filter, kernel 3
    cnn_model.add(Conv1D(filters=32, kernel_size=(9,), strides=1, padding='same',
                         activation=tf.keras.layers.LeakyReLU(alpha=0.001)))
    cnn_model.add(Dropout(0.5))
   

    #Add a noise layer to make the model more robust and help predictions with noise
    cnn_model.add(GaussianNoise(0.1))
    
    #Flatten the output
    cnn_model.add(Flatten())
    
   
    cnn_model.add(Dense(units = 128, activation=tf.keras.layers.LeakyReLU(alpha=0.001)))
    
    #Softmax as last layer with 4 outputs
    cnn_model.add(Dense(units = 4, activation='softmax'))
    
    #compile model
    cnn_model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
    
    #actually train the model
    momo=cnn_model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        class_weight=weights)
    
    #save the model
    # serialize model to JSON
    model_json = cnn_model.to_json()

    with open(model_name+'.json', "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    cnn_model.save_weights(model_name+'.h5')
    
    #its dumb, but we need to save some extra configurations XDD
    np.save(model_name,[num_HB,opt_length])