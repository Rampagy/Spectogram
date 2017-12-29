import AudioProcessing as ap
import os
import scipy.io.wavfile
import matplotlib.pyplot as plt

for (dirpath, dirnames, audio_names) in os.walk(os.path.join('urban_audio', '')):
    for audio_name in audio_names:
        print('\n' + audio_name)
        # Convert audio to image
        picture = ap.Picturize_Audio(os.path.join('urban_audio', audio_name))

        # Save image to file:
        image_name = os.path.join('Images', audio_name.split('.')[0]) + '.png'
        scipy.misc.imsave(image_name, picture)
        print(image_name)
