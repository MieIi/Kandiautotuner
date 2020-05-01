# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 19:37:54 2020

@author: antti
"""

import librosa
import numpy as np
import scipy.io.wavfile
import random
import pandas as pd
import math


into = "E:/Kandidata/Intonation/intonation.csv"

csv = pd.read_csv(into)


# Gets the 30 to 90 second part of the given file
def getPartial(filepath):
    start_time = 30
    end_time = 90
        
    try:
        audio, fs = librosa.load(path=filepath, sr=22050)
    except Exception as e:
        fs, audio_ro = scipy.io.wavfile.read(filepath)
        audio = np.copy(audio_ro) / 32767
    if fs != 22050:
        print("incorrect fs")
        return None
    audio = np.array(audio[start_time*fs:end_time*fs], dtype=np.float32)
    #librosa.output.write_wav("test_partial.wav", audio, sr=22050)
    return audio

# Creates a detuned version of the audio using the the give note indices
def createDetuned(audio, notes):  
    shifts = []
    # Using silence to parse notes
    # Shift each note
    for note in notes:
        start_time = note[0]
        end_time = note[1]
        note = np.array(audio[start_time:end_time])
        shift_amount = random.uniform(-0.8, 0.8)
        shifted = librosa.effects.pitch_shift(note, sr=22050, n_steps=shift_amount)
        shifts.append(-shift_amount)
        audio[start_time:end_time] = shifted
    return audio, shifts

# Splits a CQT of a song into CQTs of the notes
def cqtToNotes(cqt, notes):
    cqt_notes = []
    shift_insertable_indices = []
    i = 0
    for note in notes:
        start = int(note[0]/256)
        end = int(note[1]/256)
        if end - start > 150:
            amount_of_splits = int(math.ceil((end-start)/150))
            #print("we split here at note " + str(i) + " " + str(amount_of_splits) + " times")
            shift_insertable_indices.append((i, amount_of_splits))
            cqt_note = cqt[:, start:end]
            for j in range(0, amount_of_splits):
                cqt_part = cqt_note[:, 150*j:150*(j+1)]
                cqt_notes.append(cqt_part)
        else:
            cqt_note = cqt[:, start:end]
            cqt_notes.append(cqt_note)
        i = i+1
    return cqt_notes, shift_insertable_indices
    
# Pads CQT notes
def padNotes(notes):
    padded = []
    for arr in notes:
        npad = ((0,0), (0, 150-arr.shape[1]))
        pad_arr = np.pad(arr, pad_width=npad, mode='constant', constant_values=0)
        padded.append(pad_arr)
    padded = np.array(padded)
    
    return padded

# Saves arrays containing CQT notes to disk
def saveNotesToDisk(start, end):
    into = "E:/Kandidata/Intonation/intonation.csv"
    csv = pd.read_csv(into)
    d_notes = []
    b_notes = []
    all_shifts = []
    for i in range(start, end):    
        try:
            perf = csv.performance_id[i]
            path = "E:/Kandidata/Intonation/vocal_tracks/" + perf  + ".m4a"
            audio = getPartial(path)
            notes = librosa.effects.split(audio, top_db=25)
            
            # Create the detuned audio and cqt
            detuned, shifts = createDetuned(audio, notes)
            normalized_detuned = (0.01*detuned)/np.std(detuned[np.abs(detuned > 0.00001)])
            detuned_cqt = np.abs(librosa.core.cqt(normalized_detuned, fmin=125, hop_length=256, n_bins=576, bins_per_octave=96))    
            detuned_notes, insert_shifts = cqtToNotes(detuned_cqt, notes)
            
            # Get the backing track cqt
            back_cqt_path = "E:/Kandidata/Intonation/backing_features/back_cqt/" + perf  + ".npy"
            back_cqt = np.load(back_cqt_path)
            back_notes, redundant = cqtToNotes(back_cqt, notes)
            
            for i in insert_shifts:
                for j in range(1, i[1]):
                    shifts.insert(i[0]+j, shifts[i[0]])
                
            for note in detuned_notes:
               d_notes.append(note)
            for note in back_notes:
                b_notes.append(note)
            for shift in shifts:
                all_shifts.append(shift)
        except:
            continue
    # Pad notes
    d_padded= padNotes(d_notes)
    b_padded = padNotes(b_notes)
    # Save the subset to disk
    name = str(start) + '_' + str(end)
    np.save("Test set/Detuned/" + name + ".npy", d_padded)
    np.save("Test set/Backing/" + name + ".npy", b_padded)
    np.save("Test set/Shifts/" + name + ".npy", np.array(all_shifts))
    
# Saves multiple sets of CQT notes to disk for quicker loading  
def doBigSaves():
    indexes = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    for i in range(0, 10):
        start = indexes[i]
        end = indexes[i+1]
        saveNotesToDisk(start, end)

# Gets a test sample for testing the tuning
def getValidationSample():
    into = "E:/Kandidata/Intonation/intonation.csv"
    csv = pd.read_csv(into)
    perf = csv.performance_id[4000]
    path = "E:/Kandidata/Intonation/vocal_tracks/" + perf  + ".m4a"
    audio = getPartial(path)
    notes = librosa.effects.split(audio, top_db=25)
    librosa.output.write_wav("pred_test/truth.wav", audio, sr=22050)        
    # Create the detuned audio and cqt
    detuned, shifts = createDetuned(audio, notes)
    librosa.output.write_wav("pred_test/detuned.wav", detuned, sr=22050)  
    normalized_detuned = (0.01*detuned)/np.std(detuned[np.abs(audio > 0.00001)])
    detuned_cqt = np.abs(librosa.core.cqt(normalized_detuned, fmin=125, hop_length=256, n_bins=576, bins_per_octave=96))    
    detuned_notes, insert_shifts = cqtToNotes(detuned_cqt, notes)
           
    # Get the backing track cqt
    back_cqt_path = "E:/Kandidata/Intonation/backing_features/back_cqt/" + perf  + ".npy"
    back_cqt = np.load(back_cqt_path)
    back_notes, redundant = cqtToNotes(back_cqt, notes)
            
    for i in insert_shifts:
        for j in range(1, i[1]):
            shifts.insert(i[0]+j, shifts[i[0]])
    detuned_pad = padNotes(detuned_notes)          
    back_pad = padNotes(back_notes) 
    return detuned_pad, back_pad, np.array(shifts), notes


   
 
# Saves arrays containing CQT notes to disk
def gatherTestSet():
    into = "E:/Kandidata/Intonation/intonation.csv"
    csv = pd.read_csv(into)
    d_notes = []
    b_notes = []
    all_shifts = []
    for i in range(3000, 3050):    
        try:
            perf = csv.performance_id[i]
            path = "E:/Kandidata/Intonation/vocal_tracks/" + perf  + ".m4a"
            audio = getPartial(path)
            notes = librosa.effects.split(audio, top_db=25)
            
            # Create the detuned audio and cqt
            detuned, shifts = createDetuned(audio, notes)
            normalized_detuned = (0.01*detuned)/np.std(detuned[np.abs(detuned > 0.00001)])
            detuned_cqt = np.abs(librosa.core.cqt(normalized_detuned, fmin=125, hop_length=256, n_bins=576, bins_per_octave=96))    
            detuned_notes, insert_shifts = cqtToNotes(detuned_cqt, notes)
            
            # Get the backing track cqt
            back_cqt_path = "E:/Kandidata/Intonation/backing_features/back_cqt/" + perf  + ".npy"
            back_cqt = np.load(back_cqt_path)
            back_notes, redundant = cqtToNotes(back_cqt, notes)
            
            for i in insert_shifts:
                for j in range(1, i[1]):
                    shifts.insert(i[0]+j, shifts[i[0]])
                
            for note in detuned_notes:
               d_notes.append(note)
            for note in back_notes:
                b_notes.append(note)
            for shift in shifts:
                all_shifts.append(shift)
        except:
            continue
    # Pad notes
    d_padded= padNotes(d_notes)
    b_padded = padNotes(b_notes)
    # Save the subset to disk
    np.save("actual_test/detuned.npy", d_padded)
    np.save("actual_test/backing.npy", b_padded)
    np.save("actual_test/shifts.npy", np.array(all_shifts))
    

