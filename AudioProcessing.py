import os
import scipy.io.wavfile
from scipy.fftpack import fft
import numpy as np
import matplotlib.pyplot as plt


def Picturize_Audio(sound_files_dir):
    # read audio samples
    input_data = scipy.io.wavfile.read(sound_files_dir)
    # if it has 2 or more channels average them together
    ave_audio = np.zeros(shape=input_data[1].shape[0])
    if len(input_data[1].shape)>=2:
        ave_audio = np.mean(input_data[1], axis=1)
        print('Mutliple Channels')
    else: # it has one channel so don't average them
        ave_audio = input_data[1]
        print('Single Channel')

    photo_max_freq = 2500  # Hz
    photo_max_res_freq = 50 # Hz/pixel
    photo_height = int(photo_max_freq/photo_max_res_freq) # in pixels

    step = 0.1             # seconds / pixel
    photo_width = int(5.0/step) # in pixels

    sample_rate = input_data[0] # in Hz
    nyquist_freq = sample_rate/2.0 # in Hz

    song_len = len(ave_audio) / sample_rate # in seconds
    song_width = int(song_len/step) # in pixels

    fft_sects = np.zeros(shape=(photo_height, photo_width))

    # divide into step second sections
    for i in range(0, min(song_width, photo_width)):
        start = int((i*step)*sample_rate)
        stop = int(((i+1)*step)*sample_rate)

        # perform fft and only take the real frequencies
        sect_fft = np.abs(fft(ave_audio[start:stop]))

        # only use the first half of the fft (second half is redundant)
        sect_fft = sect_fft[:int(len(sect_fft)/2)]

        # frequency step spacings
        freq_step = len(sect_fft)/photo_height

        for j in range(0, photo_height - 1):
            start1 = int(j*freq_step)
            end1 = int((j+1)*freq_step)

            pixel_fft = np.mean(sect_fft[start1:end1])
            fft_sects[j, i] = pixel_fft

    # normalize to between 0 and 1
    fft_sects /= np.amax(fft_sects.flatten())
    return fft_sects

"""
# Create picture visualization:
panda_pict = Picturize_Audio("Desiigner_Panda.wav")

# Save image to file:
scipy.misc.imsave('Images/Desiigner_Panda.png', panda_pict)

# Read image back into numpy array:
panda_pict = scipy.misc.imread('Images/Desiigner_Panda.png')

# Plot Image:
plt.ylabel("Frequency, (10 Hz per pixel)")
plt.xlabel("Song Time, (0.5 sec per pixel)")
plt.title("Desiigner_Panda")

plt.imshow(panda_pict)
plt.show()
"""
