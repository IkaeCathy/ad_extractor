
from tesserocr import PyTessBaseAPI
import pytesseract
import sys
import os
import cv2
from PIL import Image
import string
import re
import numpy as np
from nltk.corpus import stopwords
import editdistance
import splitter
import enchant
from nltk.corpus import stopwords

def filter_stopwords(text,stopword_list):
    '''normalizes the words by turning them all lowercase and then filters out the stopwords'''
    words=[w.lower().split(',') for w in text if w]
    words = [item for sublist in words for item in sublist]
    #normalize the words in the text, making them all lowercase
    stopword_list = [stopword.replace('-', '') for stopword in stopword_list]
    stopword_list = [stopword.replace('\'', '') for stopword in stopword_list]
    #filtering stopwords
    filtered_words = [word for word in words if word not in stopword_list  and len(word) > 1 ]
    filtered_words.sort() #sort filtered_words list
    return filtered_words

def get_stopwords():
    eng_stop_words = set(stopwords.words('english'))
    fra_stop_words = set(stopwords.words('french'))
    deu_stop_words = set(stopwords.words('german'))
    stop_words_combined = eng_stop_words.union(fra_stop_words).union(deu_stop_words)
    return  stop_words_combined

def preprocess(image):
    if image.shape[0] < 200 and image.shape[1] >= 300:  # height
        image = cv2.resize(image, (image.shape[1] * 3, image.shape[0] * 3))

    elif image.shape[1] < 300 and image.shape[0] <= 800:  # width
        image = cv2.resize(image, (image.shape[1] * 3, image.shape[0] * 3))

    elif image.shape[0] < 300:  # height
        image = cv2.resize(image, (image.shape[1] * 2, image.shape[0] * 2))

    elif image.shape[1] < 300:  # width
        image = cv2.resize(image, (image.shape[1] * 2, image.shape[0] * 2))

    
    # multiple thresholding
    adaptive_high = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 403, 50)
    adaptive_low = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 121, 19)

    _, hard_ivs = cv2.threshold(image, 163, 255, cv2.THRESH_BINARY_INV)
    _, mid_ivs = cv2.threshold(image, 130, 255, cv2.THRESH_BINARY_INV)
    _, low_ivs = cv2.threshold(image, 109, 255, cv2.THRESH_BINARY_INV)
  

    # convert to pill imges
    adaptive_high = Image.fromarray(adaptive_high, mode='L')
    adaptive_low = Image.fromarray(adaptive_low, mode='L')
    hard_ivs = Image.fromarray(hard_ivs, mode='L')
    mid_ivs = Image.fromarray(mid_ivs, mode='L')
    low_ivs = Image.fromarray(low_ivs, mode='L')
    raw = Image.fromarray(cv2.GaussianBlur(image, (1, 1), 0), mode='L')
    # ad_ivs = Image.fromarray(adaptive_high_inv)

    pill_images = [adaptive_high,
                   adaptive_low,
                   hard_ivs,
                   mid_ivs,
                   low_ivs,
                   raw
                   # ad_ivs
                   ]
    return pill_images

def find_exemplers(image_list):
        results_r = ' '
        reward = 0
        with PyTessBaseAPI(lang='deu+fra+eng') as api:
        #with PyTessBaseAPI(oem=OEM.LSTM_ONLY, psm=PSM.AUTO_OSD, lang='deu+fra+eng') as api:
            for image in image_list:
                #image = Image.open(image)
                api.SetImage(image)
                out = api.GetUTF8Text()
                if len(out) > 1:
                    results_r += out
        results_r = " ".join(results_r.split())
       
        # results_r = results_r.replace('”','')
        translator = str.maketrans('', '', string.punctuation)
        results_r = results_r.translate(translator)  # remove puntuations
        #print("results_r =", results_r)
        ## todo make cleaning
        results_r = re.sub(r'(?<!\S)[^\s\w]+|[^\s\w]+(?!\S)', ' ', results_r)
        results_r = results_r.replace('—','-')
        #results_r = re.sub(r'[^\x00-\x7F]+', '', results_r) #non-ascii
        

        words = results_r.split(' ')  # Replace this line
        word_c = [word.upper() for word in words if
                  len(word) >= 2 and (word[1:].islower() or word[1:].isupper())]  # good words
        return word_c


image_list=["/Users/catherine/Desktop/NLP/christoph/R_D_keyword_extractor/Main/data/nondata/8.JPG"]
image =cv2.imread("/Users/catherine/Desktop/NLP/christoph/R_D_keyword_extractor/Main/data/nondata/8.JPG", 0)

pill_images = preprocess(image)
keys = find_exemplers(pill_images)
#print("keys =",set(keys))
keys =set(keys)
stop_words = get_stopwords()
print(set(filter_stopwords(keys, stop_words)))
bad, corrected = set(), set()
d_en = enchant.Dict('en_us-large')
d_fr = enchant.Dict('fr')
d_de = enchant.Dict('de_de')
for i in keys:
        out_de = ([i if d_de.check(i)==True else bad.add(i)])
##        out_fr = set([word if len(splitter.split(word, 'fr')) > 0 else bad.add(word) for word in keyword_list])
##        out_en = set([word if len(splitter.split(word, 'en_us-large')) > 0 else bad.add(word) for word in keyword_list])
##        out = out_de.union(out_en).union(out_fr)
        print(out_de)
        bad = bad - set(out_de)
print(bad)
print(keys-bad)



