'''
this is code for using tesseract OCR engines
using tesserocr python wrapper and for use in pycharm
'''
import concurrent.futures
import string
from collections import defaultdict
import cv2
import editdistance
import numpy as np
import re
import sklearn.cluster
from PIL import Image
from tesserocr import PyTessBaseAPI, OEM, PSM
import matplotlib.pyplot as plt

def preprocess(image):
    
    

    # if image.shape[1] <= 200:
    #     basewidth = 300
    #     wpercent = (basewidth / float(image.shape[0]))
    #     hsize = int((float(image.shape[1]) * float(wpercent)))
    #     image = cv2.resize(image,(basewidth, hsize))
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

    # adaptive_high_inv = cv2.adaptiveThreshold(image, 250, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 111, 0)
    # adaptive inverse works when hard is too high
    _, hard_ivs = cv2.threshold(image, 163, 255, cv2.THRESH_BINARY_INV)
    _, mid_ivs = cv2.threshold(image, 130, 255, cv2.THRESH_BINARY_INV)
    _, low_ivs = cv2.threshold(image, 109, 255, cv2.THRESH_BINARY_INV)
    #cv2.imshow('raw', image)
    #cv2.imshow('low',adaptive_low)
    # cv2.imshow('adlow',adaptive_high_inv)
    # cv2.imshow('mid',mid_ivs)
    # cv2.imshow('high',hard_ivs)
    # cv2.waitKey()

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

from tesserocr import PyTessBaseAPI
def find_exemplers(image_list):
    try:
        results_r = ' '
        reward = 0
        clusters = defaultdict(str)
        with PyTessBaseAPI(oem=OEM.DEFAULT, psm=PSM.AUTO_OSD,lang='deu+fra+eng') as api:
        #with PyTessBaseAPI(oem=OEM.LSTM_ONLY, psm=PSM.AUTO_OSD, lang='deu+fra+eng') as api:
            for image in image_list:
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
        words = [x.upper() for x in words if
                 len(x) > 4] + word_c+ word_c  # all other words should be greater than 6 words to be included with good words
        words = np.asarray(words)  # So that indexing with a list will work
        lev_similarity = -.99* np.array([[editdistance.eval(w1, w2) for w1 in words] for w2 in words])
        affprop = sklearn.cluster.AffinityPropagation(affinity="precomputed", damping=0.5)#, random_state=0)
        affprop.fit(lev_similarity)

        for cluster_id in np.unique(affprop.labels_):
            exemplar = words[affprop.cluster_centers_indices_[cluster_id]]
            cluster = np.unique(words[np.nonzero(affprop.labels_ == cluster_id)])
            cluster_str = ", ".join(cluster)
            if len(exemplar) > 1 and exemplar.isalpha():
                clusters[exemplar] = cluster_str

            # print(" - *%s:* %s" % (exemplar, cluster_str))
        # output_data.append([ad, ','.join(list(clusters.keys()))])
        #keywords =  ','.join(str(clusters.keys()))
        #split compound stopwords

    except Exception as exc:
        
        #print('tesser generated an exception: %s' %  exc)
        # save some default to csv
        # todo move file to a skew im memory list, and run with skew(use folder name for logic)
        # output_data.append([ad, '--'])
        pass
    #return [ad, ','.join(list(clusters.keys()))] #with file location
    return [(','.join(list(clusters.keys())))]# only words

def read_keys(image): #recieves a numpy array
    # recieve a  pil and process
    #plt.imshow(image)
    #plt.show()
    image = np.uint8(image * 209)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #image = Image.fromarray(image)
    #dsimage = deskew(image)
    pill_images = preprocess(image)
    #print(" pill_images = ",  pill_images[0])#images currently has no content shown....
    #pill_images1 = preprocess(dsimage)
    keys = find_exemplers(pill_images)
    print("keys =",keys)
    return keys


# executor = ThreadPoolExecutor(max_workers=10)
def execute_images(tensor):
    images_list = list(tensor)
    output_data = []

    #loop through batch

    # We can use a with statement to ensure threads are cleaned up promptly
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # Start the load operations and mark each future with its URL
        future_to_list = {executor.submit(read_keys, image): image for image in images_list}
        for future in concurrent.futures.as_completed(future_to_list):
            adwords = future_to_list[future]
            try:
                data = future.result()
            except IOError: #todo should be moved to the frontend
                print('not in the right format')
            except Exception as exc:
                print('%r generated an exception: %s' % (adwords, exc))
            else:
                # print('keys is =  %r ' %data)
                output_data.append(','.join(data)) #list of list of keywords
    return  output_data

        #end here return output to the model for loss

#         with open(str(path) + '.csv', "w") as output:
#             # csv.writer("Image name, Keywords\n")
#             writer = csv.writer(output, lineterminator='\n')
#             writer.writerows(output_data)
#         output_data.clear()
