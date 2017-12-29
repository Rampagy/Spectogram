import scipy.io.wavfile
import os
import numpy as np
#import matplotlib.pyplot as plt

def ReadImage():
    image_dict = {}

    for (dirpath, dirnames, image_names) in os.walk(os.path.join('Images', '')):
        # Initialize an array with as many lines as images and length of 50*50
        # because the pictures are 50 pixels by 50 pixels
        audio_images = np.zeros(shape=(len(image_names), 50*50))

        for image_name in image_names:
            # Read image back into numpy array:
            audio_pict = scipy.misc.imread(os.path.join('Images', image_name)).flatten()/255

            truth_id = image_name.split('-')
            image_dict[image_name] = (audio_pict, int(truth_id[1]))

    return image_dict

"""
# Plot Image:
plt.ylabel("Frequency, (10 Hz per pixel)")
plt.xlabel("Song Time, (0.5 sec per pixel)")
plt.title("Desiigner_Panda")

plt.imshow(panda_pict)
plt.show()
"""
