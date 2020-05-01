"""
Created on Thu Mar  5 12:48:55 2020

@author: Antti
"""

import numpy as np
import librosa
import librosa.display
from matplotlib import pyplot as plt
import pandas as pd
from get_dataset import getPartial, padNotes, cqtToNotes, createDetuned



def exampleCQTs():
    into = "E:/Kandidata/Intonation/intonation.csv"
    csv = pd.read_csv(into)
    
    perf = csv.performance_id[0]
    path = "E:/Kandidata/Intonation/vocal_tracks/" + perf  + ".m4a"
    audio = getPartial(path)
    notes = librosa.effects.split(audio, top_db=25)
    
    normalized_audio = (0.01*audio)/np.std(audio[np.abs(audio > 0.00001)])
    cqt = np.abs(librosa.core.cqt(normalized_audio, fmin=125, hop_length=256, n_bins=576, bins_per_octave=96))    
    truth_notes = padNotes(cqtToNotes(cqt, notes)[0])
            
    # Create the detuned audio and cqt
    detuned, shifts = createDetuned(audio, notes)
    normalized_detuned = (0.01*detuned)/np.std(detuned[np.abs(detuned > 0.00001)])
    detuned_cqt = np.abs(librosa.core.cqt(normalized_detuned, fmin=125, hop_length=256, n_bins=576, bins_per_octave=96))    
    detuned_notes = padNotes(cqtToNotes(detuned_cqt, notes)[0])
            
    # Get the backing track cqt
    back_cqt_path = "E:/Kandidata/Intonation/backing_features/back_cqt/" + perf  + ".npy"
    back_cqt = np.load(back_cqt_path)
    back_notes = padNotes(cqtToNotes(back_cqt, notes)[0])
    
    
    
    
    # Plot example cqts
    db1 = librosa.core.power_to_db(cqt)
    #plt.imshow(db, aspect='auto', origin='lower')
    db2 = librosa.core.power_to_db(detuned_cqt)
    #plt.imshow(db, aspect='auto', origin='lower')
    db3 = librosa.core.power_to_db(back_cqt)
    #plt.imshow(db, aspect='auto', origin='lower')
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(10,4))
    fig.add_subplot(111, frameon=False, xticks=[], yticks=[])
    #fig.suptitle('Constant-Q transforms of a whole song')
    
    ax1.imshow(db1, aspect='auto', origin='lower')
    ax1.title.set_text('In-tune')
    ax2.imshow(db2, aspect='auto', origin='lower')
    ax2.title.set_text('Out of tune')
    ax3.imshow(db3, aspect='auto', origin='lower')
    ax3.title.set_text('Backing track')
    fig.text(0.5, 0.04, 'Frames', ha='center', va='center')
    fig.text(0.05, 0.5, 'Frequency bins', ha='center', va='center', rotation='vertical')
    
    #plt.xlabel("Frames")
    #plt.ylabel("Frequency bins")
   # plt.colorbar(format='%+2.0f dB')

    plt.figure()

    ind = 2
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=True)
    fig.add_subplot(111, frameon=False, xticks=[], yticks=[])
    # Plot example cqts
    db4 = librosa.core.power_to_db(truth_notes[ind])
    db5 = librosa.core.power_to_db(detuned_notes[ind],)
    db6 = librosa.core.power_to_db(back_notes[ind])
    ax1.imshow(db4, aspect='auto', origin='lower')
    ax1.title.set_text('In-tune')
    ax2.imshow(db5, aspect='auto', origin='lower')
    ax2.title.set_text('Out of tune')
    ax3.imshow(db6, aspect='auto', origin='lower')
    ax3.title.set_text('Backing track')
    fig.text(0.5, 0.04, 'Frames', ha='center', va='center')
    fig.text(0.05, 0.5, 'Frequency bins', ha='center', va='center', rotation='vertical')
   # plt.title("Constant Q-transform of a note")
    #plt.colorbar(format='%+2.0f dB')
    
exampleCQTs()
    
    
"""
detuned = np.load("Test set/Detuned/101304813_2186385057.npy")     
back = np.load("Test set/Backing/101304813_2186385057.npy")
truth = np.load("Test set/Ground truth/101304813_2186385057.npy")

# Plot example cqts
db = librosa.core.power_to_db(detuned, ref=1.0)
plt.imshow(db, aspect='auto', origin='lower')
plt.xlabel("Frames")
plt.ylabel("Frequency bins")
plt.title("Input vocal")
plt.colorbar(format='%+2.0f dB')

plt.figure()

db = librosa.core.power_to_db(back, ref=1.0)
plt.imshow(db, aspect='auto', origin='lower')
plt.xlabel("Frames")
plt.ylabel("Frequency bins")
plt.title("Backing track")
plt.colorbar(format='%+2.0f dB')

plt.figure()

db = librosa.core.power_to_db(truth, ref=1.0)
plt.imshow(db, aspect='auto', origin='lower')
plt.xlabel("Frames")
plt.ylabel("Frequency bins")
plt.title("Target")
plt.colorbar(format='%+2.0f dB')

plt.show()



# Plot zoomed in
db = librosa.core.power_to_db(detuned[80:200:,0:200], ref=1.0)
plt.imshow(db, aspect='auto', origin='lower')
plt.xlabel("Frames")
plt.ylabel("Frequency bins")
plt.title("Input vocal")
plt.colorbar(format='%+2.0f dB')

plt.figure()


db = librosa.core.power_to_db(truth[80:200:,0:200], ref=1.0)
plt.imshow(db, aspect='auto', origin='lower')
plt.xlabel("Frames")
plt.ylabel("Frequency bins")
plt.title("Target")
plt.colorbar(format='%+2.0f dB')

plt.show()
"""




