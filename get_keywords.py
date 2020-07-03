'''
script for using model to extract keywords
'''
##import sys
##sys.path.append("/anaconda3/lib/python3.7/site-packages")

import concurrent.futures
import csv
import os
import pathlib
import torch
from PIL import Image
from configargparse import YAMLConfigFileParser
from torchvision.transforms import functional
#from yaap import ArgParser
from collections import Counter
from nltk.corpus import stopwords
import enchant
import editdistance
import splitter

from train_model import model,args
from tessocr import read_keys




def get_keys_from_ad(ad_path):

    image = Image.open(ad_path)
    image = image.convert('RGB')
    #make transforms
    image = functional.to_tensor(image)
    image = functional.normalize(image, mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    image = image.unsqueeze(0)
    #call model to modify the image here
    #output of the image is a numpy array(nd array)
    model.eval()
    with torch.no_grad():
        #out = model(image.cuda())
        out = model(image.cpu())
        nd_out = out.clone()
        nd_out = nd_out.detach().cpu().squeeze(0).numpy().transpose(1, 2, 0)
        # plot_grayscale_image(image.squeeze(0))
        # plot_numpy_image(nd_out)
        tesser_out = read_keys(nd_out)
        #print(tesser_out)
    return [ad_path, str(tesser_out)]


# executor = ThreadPoolExecutor(max_workers=10)
output_data = [] # [['Image_loc', "Keywords"],]


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

#post_process(output_data, args.destination)
def post_process(out_list, save_to):
    stop_words = get_stopwords()
    bad, corrected = set(), set()
    d_en = enchant.Dict('en_us-large')
    d_fr = enchant.Dict('fr')
    d_de = enchant.Dict('de_de')
    suggested = []
    for i in out_list:
        image_name = i[0].split('/')[-1]#*************split the image with a forward slash instead of the backward slash
        print(image_name)
        keyword_list = i[1][2:-2].split(',')  # get very sure words from the dictionary
        # keyword_list = keywords[i[0]].split(',')
        out_de = set([word if len(splitter.split(word, 'de_de')) > 0 else bad.add(word) for word in keyword_list])#change spliter to str
        out_fr = set([word if len(splitter.split(word, 'fr')) > 0 else bad.add(word) for word in keyword_list])
        out_en = set([word if len(splitter.split(word, 'en_us-large')) > 0 else bad.add(word) for word in keyword_list])
        out = out_de.union(out_en).union(out_fr)
        #print("bad==",bad,len(bad))
        bad = bad - out
        if "" in out:
            out = {"NO KEY WORDS EXTRACTED"}
        
        
        #lang = ''
        # bad = set(filter_stopwords(bad, stop_words_combined))
        # make suggestions in the 3 langs
        if bad:
            #bad = set(filter_stopwords(bad, stop_words))
            # suggested = [item for sublist in suggested for item in sublist]
            for i in bad:
              #print("i==",len(i))
              if len(i)>0:#+++++++++++++++++++++++++++++++
                suggested = d_de.suggest(i)
                suggested1 = d_fr.suggest(i)
                suggested2 = d_en.suggest(i)
                dis = [editdistance.eval(i, s.replace(' ', '').replace('-', '')) for s in suggested]
                dis1 = [editdistance.eval(i, s.replace(' ', '').replace('-', '')) for s in suggested1]
                dis2 = [editdistance.eval(i, s.replace(' ', '').replace('-', '')) for s in suggested2]
                # dis1 = ([len(set(i) & set(s))/len(set(i).union(set(s.replace(' ','')))) for s in suggested])
                try:
                    d = {len(dis) + sum(dis): min(dis), len(dis1) + sum(dis1): min(dis1), len(dis2) + sum(dis2): min(dis2)}
                    d = sorted(d.items(), key=lambda x: (x[1], -x[0]))
                    # m = min(d, key=d.get)
                    di = [len(dis2) + sum(dis2), len(dis1) + sum(dis1), len(dis) + sum(dis)]
                    sug = [suggested2, suggested1, suggested][di.index(d[0][0])]
                    dd = [dis2, dis1, dis][di.index(d[0][0])]
                    if dd and min(dd) < 4:
                        if min(dd) == 0 or min(dd) < 3 and len(sug[dd.index(min(dd))].replace(' ', '')) > 3 and \
                                dd.count(Counter(dd).most_common(1)[0][0]) / len(dd) < .5 and sum(dd) / len(dd) < 3.5:
                            corrected = sug[dd.index(min(dd))].replace(' ', '') \
                                if len(sug[dd.index(min(dd))].partition(' ')[0]) <= 2 \
                                else ','.join(sug[dd.index(min(dd))].split())

                        else:
                            corrected = ''
                    elif len(i) >= 4 and min(dd) < 4 and dd.count((min(dd))) / len(dd) < .5:
                        corrected = i
                        # out.add(corrected)
                    else:
                        corrected = ''

                except:
                    corrected = ''
                    pass
              else:#*************
                    corrected="NO KEY WORDS EXTRACTED"#+++++++++++
              out.add(corrected)
               
              #print('%s --> %s' % (i, corrected))
            bad.clear()

        corrected = ""
        
        out = set(filter_stopwords(out, stop_words))
        if len(out)<1:
            out={"NO KEY WORDS EXTRACTED"}
             
        with open((r'%s/%s.txt'%(save_to, image_name)), "w", encoding='utf8') as fp:  # unpacking
            #print("OUT==", out)
            for s in out:
                fp.write(str(s).upper() + '\n' )


def run():

    for path, subdirs, files in os.walk(args.source_folder):
    # We can use a with statement to ensure threads are cleaned up promptly
        print("image files==", files)
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            # Start the load operations and mark each future with its URL
            future_to_list = {executor.submit(get_keys_from_ad, str(pathlib.Path(path).joinpath(ad))): ad for ad in files}
            for future in concurrent.futures.as_completed(future_to_list):
                adwords = future_to_list[future]
                try:
                    data = future.result()
                except IOError: #todo should be moved to the frontend
                    print('Input not in the right format')
                except Exception as exc:
                    print('%r generated an exception: %s' % (adwords, exc))
                else:
                    # print('keys is =  %r ' %data)
                    output_data.append(data)
            # with open(str(path) + '.csv', "w") as output:
            #     #csv.writer("Image_loc, Keywords\n")
            #     writer = csv.writer(output, lineterminator='\n')
            #     writer.writerows(output_data)
            # output_data.clear()

    post_process(output_data, args.destination)
if __name__ == '__main__':
    run()
