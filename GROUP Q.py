#BUKENYA CALVIN 2024/BSE/049/PS
#IMAMUT JULIAN 2024/BSE/066/PS
#MUYAMBI STEPHEN 2024/BSE/125/PS
#NUWAHEREZA SYLAS 2024/BSE/157/PS
#NABULYA SHADIA 2024/BSE/134/PS
import librosa
import numpy as np
import scipy.signal as sig
from collections import Counter
import matplotlib.pyplot as plt
import psola  # Ensure you have a working PSOLA vocoder module
import soundfile as sf

# Load audio
file_path = '/home/calvin/Desktop/output1.wav'
y, sr = librosa.load(file_path, sr=None)



def freq_to_note(freq):
    A4 = 440.0#starndard pitch 
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    if freq <= 0:
        return None
    semitones = int(round(12 * np.log2(freq / A4)))#calculate the pitch class for any given frequency
    note_index = (semitones + 9) % 12  # A4 is index 9
    return note_names[note_index]

frame_size = 2048#defines how many audio samples to be analyzed at once
hop_length = 512#how far to slide the window forward each time
note_names = []
#analyse the entire audio file chunck by chunk ,deetect most dominant frequency and map them into musical notes
for i in range(0, len(y) - frame_size, hop_length):
    frame = y[i:i + frame_size]
    spectrum = np.fft.fft(frame)#apply FFT to each frame(turn time domain audio to frequency domain)
    freqs = np.fft.fftfreq(len(frame), 1/sr)
    magnitudes = np.abs(spectrum[:len(freqs)//2])
    freqs = freqs[:len(freqs)//2]
    max_index = np.argmax(magnitudes)#find dominant frequency
    dominant_freq = freqs[max_index]
    note = freq_to_note(dominant_freq)#convert dominant frequency to musical note
    if note:
        note_names.append(note)#store these notes

note_counter = Counter(note_names)#store detected notes
in_tune_notes = [note for note, count in note_counter.items() if count > 5]#identify in tune notes

print("Detected in-tune notes:", in_tune_notes)

note_degrees = np.array([librosa.note_to_midi(n) % 12 for n in in_tune_notes])

# Autotune using those detected in-tune notes ===

def correct(f0, allowed_degrees):
    if np.isnan(f0):
        return np.nan
    midi_note = librosa.hz_to_midi(f0)
    degree = midi_note % 12
    closest_id = np.argmin(np.abs(allowed_degrees - degree))
    corrected_degree = allowed_degrees[closest_id]
    degree_diff = corrected_degree - degree
    midi_note += degree_diff
    return librosa.midi_to_hz(midi_note)

def correct_pitch(f0, allowed_degrees):
    corrected_f0 = np.zeros_like(f0)
    for i in range(len(f0)):
        corrected_f0[i] = correct(f0[i], allowed_degrees)
    corrected_f0 = sig.medfilt(corrected_f0, kernel_size=11)
    corrected_f0[np.isnan(corrected_f0)] = f0[np.isnan(corrected_f0)]
    return corrected_f0

def autotune(y, sr, note_degrees):
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)

    corrected_f0 = correct_pitch(f0, note_degrees)

    #visualize
    times = librosa.times_like(f0)
    plt.figure(figsize=(12, 4))
    plt.plot(times, f0, label='Original Pitch', alpha=0.7)
    plt.plot(times, corrected_f0, label='Corrected Pitch', alpha=0.7)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Pitch Correction Visualization")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return psola.vocode(y, sample_rate=sr, target_pitch=corrected_f0,
                        fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))

# Apply autotuning
output = autotune(y, sr, note_degrees)

# Save output
sf.write('/home/calvin/Desktop/autotuned_.wav', output, sr)
print("Autotuned audio saved.")