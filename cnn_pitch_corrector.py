"""
Created on Thu Mar  5 12:48:55 2020

@author: Antti
"""


from keras.models import Model
from keras.layers import Input, Dense, Concatenate, BatchNormalization, Conv2D, Reshape, Flatten, MaxPooling2D
from keras.optimizers import Adam
from keras.backend import clear_session
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from get_dataset import getValidationSample, padNotes
import os
import librosa
import scipy


def generateBatches():
    names = os.listdir('Test set/Detuned/')[:-1]
    detuned_path = 'Test set/Detuned/'
    back_path = 'Test set/Backing/'
    shifts_path = 'Test set/Shifts/'
    #while True:
    for name in names:
        detuned = np.load(detuned_path + name)
        assert not np.any(np.isnan(detuned)), 'nan found in detuned'
        back = np.load(back_path + name)
        assert not np.any(np.isnan(back)), 'nan found in backing'
        shifts = np.load(shifts_path + name)
        assert not np.any(np.isnan(shifts)), 'nan found in shifts'
        yield [detuned, back], shifts

                

def testPrediction(model):
    # Get validation sample note cqts
    detuned_pad, back_pad, shifts, note_indices = getValidationSample()
        
    # Get predicted shifts
    corrections = model.predict([detuned_pad, back_pad])
    
    # Get detuned audio
    try:
        audio, fs = librosa.load(path="pred_test/detuned.wav", sr=22050)
    except Exception as e:
        fs, audio_ro = scipy.io.wavfile.read("pred_test/detuned.wav")
        audio = np.copy(audio_ro) / 32767
    if fs != 22050:
        print("incorrect fs")
        return None
    # Apply the predicted shifts to detuned audio
    i = 0
    for note in note_indices:
        start_time = note[0]
        end_time = note[1]
        note = np.array(audio[start_time:end_time])
        shifted = librosa.effects.pitch_shift(note, sr=22050, n_steps=corrections[i])
        audio[start_time:end_time] = shifted
        i = i + 1     
    librosa.output.write_wav("pred_test/tuned.wav", audio, sr=22050) 
    return
  


longestNote = 150

clear_session()
vocal = Input(shape=(576, longestNote), name='vocal') 
v = Reshape((576, -1, 1))(vocal)
backing = Input(shape=(576, longestNote), name='backing')
b = Reshape((576, -1, 1))(backing)
x = Concatenate()([v, b])
x = Conv2D(32, kernel_size=11, strides=1, padding='valid', activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D(2)(x)
x = Conv2D(64, kernel_size=7, strides=1, padding='valid', activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D(2)(x)
x = Conv2D(128, kernel_size=5, strides=1, padding='valid', activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D(2)(x)
x = Conv2D(128, kernel_size=3, strides=1, padding='valid', activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D(2)(x)
x = Flatten()(x)
output = Dense(1, activation='linear')(x)
model = Model(inputs=[vocal, backing], outputs=output)
my_adam = Adam(learning_rate=0.0001)
model.compile(optimizer=my_adam, loss='mean_squared_error')
model.summary()


nb_epoch=33

valid_name = os.listdir('Test set/Detuned/')[-1]
validation_set = ([np.load('Test set/Detuned/' + valid_name), np.load('Test set/Backing/' + valid_name)], np.load('Test set/Shifts/' + valid_name))

train_history = []
val_history = []

for e in range(nb_epoch):
    train_loss = []
    print("epoch %d" % e)
    for X_train, Y_train in generateBatches(): 
        train_loss = model.fit(X_train, Y_train, epochs=1, batch_size=4, verbose=1)
    train_history.append(train_loss.history['loss'])
    val_loss = model.evaluate(validation_set[0], validation_set[1])
    val_history.append(val_loss)

model.save('pitch_corrector.h5')

testPrediction(model)



# Dump the losses
import pickle
with open('train_history.p', 'wb') as f:
    pickle.dump(train_history, f)
    
with open('val_history.p', 'wb') as f:
    pickle.dump(val_history, f)
    

plt.plot(train_history, 'r')
plt.plot(val_history, 'b')
plt.legend(['Training loss', 'Validation loss'])
plt.xlabel('Epoch')
plt.ylabel('Mean-squared-error')
plt.show()


def testSet():
    test_detuned = np.load("actual_test/detuned.npy")
    test_back = np.load("actual_test/backing.npy")
    test_shifts = np.load("actual_test/shifts.npy")
    return test_detuned, test_back, test_shifts

td, tb, ts = testSet()

test_result = model.evaluate([td, tb], ts)

# Predict shifts, plot the actual and predicted to observe   
predicted_shifts = model.predict([td, tb])

"""
plt.figure()
#plt.plot(ts[:50], 'r.')
#plt.plot(predicted_shifts[:50], 'b.')
plt.plot(ts[:50].reshape(50), predicted_shifts[:50].reshape(50)
plt.legend(['Target shift', 'Predicted shift'])
plt.xlabel('Sample')
plt.ylabel('Shift amount')
plt.show()
"""


# How fucking hard can it be to plot a vertical line between two points...
# Apparently impossible, fuck this so much
from matplotlib import collections as matcoll
x = []
lines = []
for i in range(100):
    pair = [(i, ts[i]), (i, predicted_shifts[i])]
    lines.append(pair)

linecoll = matcoll.LineCollection(lines, colors='k')

fig, ax = plt.subplots()
ax.plot(ts[:100], 'r.')
ax.plot(predicted_shifts[:100], 'b.')
plt.legend(['Target shift', 'Predicted shift'])
ax.add_collection(linecoll)

