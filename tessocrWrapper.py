import argparse
import csv
import string
from collections import defaultdict
import concurrent.futures
import cv2
import editdistance
import numpy as np
import re
import sklearn.cluster
from PIL import Image
from tesserocr import PyTessBaseAPI, OEM, PSM
import os
import pathlib
from OSD import deskew


def read_keys_from_ad(nd_out):
    image = np.uint8(nd_out)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    r_image = Image.fromarray(image, mode='L')#Image.open(ad)
    #call model to modify the image here
    #output of the image is a numpy array(nd array) comes here
    #image = cv2.imread(ad, cv2.IMREAD_GRAYSCALE)  # read as grayscaled image
    #image = deskew(image)

    if image.shape[1] <= 350 and image.shape[0] <= 300:
        image = cv2.resize(image, (image.shape[1] * 3, image.shape[0] * 3))
    elif image.shape[1] < 300 and image.shape[0] <= 600:  # width
        image = cv2.resize(image, (image.shape[1] * 2, image.shape[0] * 2))
    elif image.shape[0] < 100 and image.shape[1] >= 300:  # height
        image = cv2.resize(image, (image.shape[1] * 3, image.shape[0] * 3))
    elif image.shape[0] < 100:  # height
        image = cv2.resize(image, (image.shape[1] * 3, image.shape[0] * 3))
    elif image.shape[1] < 100:  # width
        image = cv2.resize(image, (image.shape[1] * 4, image.shape[0] * 4))

    # blur image a bit, deskew
    blured_image = cv2.GaussianBlur(image, (1, 1), 0)

    # multiple thresholding
    adaptive_high = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 505, 80)
    adaptive_low = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 119, 20)

    # adaptive_high_inv = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 1719, 0)
    # adaptive inverse works when hard is too high
    _, hard_ivs = cv2.threshold(image, 240, 255, cv2.THRESH_BINARY_INV)
    _, mid_ivs = cv2.threshold(image, 132, 255, cv2.THRESH_BINARY_INV)
    _, low_ivs = cv2.threshold(image, 169, 255, cv2.THRESH_BINARY_INV)

    # convert to pill imges
    adaptive_high = Image.fromarray(adaptive_high)
    adaptive_low = Image.fromarray(adaptive_low)
    hard_ivs = Image.fromarray(cv2.GaussianBlur(hard_ivs, (1, 1), 0))
    mid_ivs = Image.fromarray(cv2.GaussianBlur(mid_ivs, (1, 1), 0))
    low_ivs = Image.fromarray(cv2.GaussianBlur(low_ivs, (3, 3), 0))

    '''#low_ivs_img.save('o.jpg', dpi=(300,300))
    #adaptive_high_invimg = Image.fromarray(adaptive_high_inv)
    # multiple thresholding
    adaptive_high_sk = cv2.adaptiveThreshold(dsk_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 505, 80)
    adaptive_low_sk = cv2.adaptiveThreshold(dsk_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 119, 20)
'''

    pill_images = [adaptive_high,
                   adaptive_low,
                   hard_ivs,
                   mid_ivs,
                   low_ivs,
                   Image.fromarray(blured_image),
                   r_image
                   ]
#     pill_images = [
#                    r_image,
#                     r_image,
#         r_image
#                    ]

    # todo make as a function call here to be used multiple times (or a class?)
    try:
        results_r = ' '
        clusters = defaultdict(str)
        with PyTessBaseAPI(oem=OEM.LSTM_ONLY, psm=PSM.AUTO_OSD, lang='deu+fra+eng') as api:
            for image in pill_images:
                api.SetImage(image)
                out = api.GetUTF8Text()
                if len(out) > 3:
                    results_r += api.GetUTF8Text()
        results_r = " ".join(results_r.split())
        # results_r = results_r.replace('”','')
        translator = str.maketrans('', '', string.punctuation)
        results_r = results_r.translate(translator)  # remove puntuations

        ## todo make cleaning
        results_r = re.sub(r'(?<!\S)[^\s\w]+|[^\s\w]+(?!\S)', ' ', results_r)
        results_r = results_r.replace('—','')
        # print(results_r)

        words = results_r.split(' ')  # Replace this line
        word_c = [word.upper() for word in words if
                  len(word) > 2 and (word[1:].islower() or word[1:].isupper())]  # good words
        words = [x.upper() for x in words if
                 len(x) > 4] + word_c  # all other words should be greater than 6 words to be included with good words
        words = np.asarray(words)  # So that indexing with a list will work
        lev_similarity = -97 * np.array([[editdistance.eval(w1, w2) for w1 in words] for w2 in words])
        affprop = sklearn.cluster.AffinityPropagation(affinity="precomputed", damping=0.5)

        affprop.fit(lev_similarity)

        for cluster_id in np.unique(affprop.labels_):
            exemplar = words[affprop.cluster_centers_indices_[cluster_id]]
            cluster = np.unique(words[np.nonzero(affprop.labels_ == cluster_id)])
            cluster_str = ", ".join(cluster)
            if len(exemplar) > 1:
                # put into dictionary where key exist
                clusters[exemplar] = cluster_str
            # print(" - *%s:* %s" % (exemplar, cluster_str))
        # output_data.append([ad, ','.join(list(clusters.keys()))])

    except:
        # save some default to csv
        # todo move file to a skew im memory list, and run with skew(use folder name for logic)
        # output_data.append([ad, '--'])
        pass

    ##todo
    return [ ','.join(list(clusters.keys()))] #with file location
    #return [ ','.join(list(clusters.keys()))]


# executor = ThreadPoolExecutor(max_workers=10)
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--source', type=str,  help="indicate location to folder where images are found"
                                                         "to be processed")
    args = parser.parse_args()
    output_data = []
    for path, subdirs, files in os.walk(args.p):

        # We can use a with statement to ensure threads are cleaned up promptly
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            # Start the load operations and mark each future with its URL
            future_to_list = {executor.submit(read_keys_from_ad, str(pathlib.Path(path).joinpath(ad))): ad for ad in files}
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
                    output_data.append(data)

            with open(str(path) + '.csv', "w") as output:
                # csv.writer("Image name, Keywords\n")
                writer = csv.writer(output, lineterminator='\n')
                writer.writerows(output_data)
            output_data.clear()
