# imports

from keras.applications.xception import Xception, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical, plot_model
from keras.layers.merge import add
from keras.models import Model, load_model
from keras.layers import Input, Dense, LSTM, Embedding, Dropout
from keras.preprocessing.text import Tokenizer
import os
from pickle import dump, load
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import argparse

from tqdm import tqdm_notebook as tqdm
tqdm().pandas()

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='Image Path')
args = vars(ap.parse_args())
img_path = args['image']

def extract_features(filepath, model):
    try:
        image = Image.open(filepath)

    except:
        print("Can't open the image. Please make sure the path and extension are correct.")
    
    image = image.resize((299,299))
    image = np.array(image)

    #force three-channel inputs
    if image.shape[2] == 4:
        image = image[..., :3]
    
    image = np.expand_dims(image, axis=0)
    image = image/127.5
    image = image - 1.0
    feature = model.predict(image)

    return feature


def word_for_id(num, tokenizer):
    for w, i in tokenizer.word_index.items():
        if i == num:
            return w
    return None


def generate_caption(model, tokenizer, pic, longest_caption):
    in_text = 'start'

    for i in range(longest_caption):
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = pad_sequences([seq], maxlen=longest_caption)

        pred = model.predict([pic, seq], verbose=0)
        pred = np.argmax(pred)

        word = word_for_id(pred, tokenizer)

        if word is None:
            break
        in_text += ' ' + word

        if word == 'end':
            break

    return in_text

longest_caption = 155
tokenizer = load(open('/home/mattiboi/Desktop/Data Science/Metis/Deep Learning/project/data/tokens.p', 'rb'))
model = load_model('/home/mattiboi/Desktop/Data Science/Metis/Deep Learning/project/models/model_24.h5')
xception_model = Xception(include_top=False, pooling='avg')

pic = extract_features(img_path, xception_model)
img = Image.open(img_path)

caption = generate_caption(model, tokenizer, pic, longest_caption)
print("\n\n")
print(caption)
plt.imshow(img)

## live captioning at the command line process created by Abhishek Sharma (https://medium.com/mlearning-ai/image-captioning-using-deep-learning-with-source-code-easy-explanation-3f2021a63f14)