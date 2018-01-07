import os
import numpy as np
import tensorflow as tf
import AudioClassification as ac
import AudioProcessing as ap

class_dict = {
    -1:'ERROR',
    0:'air_conditioner',
    1:'car_horn',
    2:'children_playing',
    3:'dog_bark',
    4:'drilling',
    5:'engine_idling',
    6:'gun_shot',
    7:'jackhammer',
    8:'siren',
    9:'street_music'}


predictions_dict = {}


# Go through each file in folder "Predict" and
# predict which classes the audio belongs to
for (dirpath, dirnames, audio_names) in os.walk(os.path.join('Predict_audio', '')):
    for audio_name in audio_names:
        audio_image = np.float32(ap.Picturize_Audio(os.path.join('Predict_audio', audio_name)))
        audio_image = np.asarray(audio_image).flatten()
        audio_image = np.reshape(audio_image, [1, 50*50])

        # Create the Estimator
        audio_classifier = tf.estimator.Estimator(model_fn=ac.cnn_model_fn, model_dir=os.path.join(os.path.dirname(os.path.realpath(__file__)), "urban_sound_model"))
        pred_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": audio_image}, num_epochs=1, shuffle=False)
        eval_results = audio_classifier.predict(input_fn=pred_input_fn)

        # extract the probabilities from the generator
        class_probabilities = next(eval_results)['probabilities']

        # find the max probability
        max_prob = -1
        position = -1

        for i in range(0, len(class_probabilities)):
            if class_probabilities[i] > max_prob:
                max_prob = class_probabilities[i]
                position = i

        predictions_dict[audio_name] = (class_dict[position], max_prob)

for key, val in predictions_dict.items():
    truth = key.split('-')
    print(key + ':')
    print('  Guess: ' + val[0])
    print('  Truth: ' + class_dict[int(truth[1])])
    print('  Probs: ' + repr(val[1]))
