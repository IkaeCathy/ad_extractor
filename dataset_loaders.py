
#load files from a csv file....

##import sys
##sys.path.append("/anaconda3/lib/python3.7/site-packages")
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])  ])

def make_square(img, min_size=600, fill_color=(0, 0, 0, 0)):
    basewidth = 600
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), Image.ANTIALIAS)
    if img.size[1] <= 350:
        img = img.resize( [int(1.5* s) for s in img.size] )
    new_im = Image.new('RGBA', (min_size, min_size), fill_color)
    new_im.paste(img, (0,0))
    return new_im

class AdvertDataset(Dataset):
    """An override of class Dataset.

    All  datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __init__(self,  path_to_csv, tranforms= None):
        #some arguments needed for music dataset to function?
        #place init of them here, like csv, folder_loc, etc
        tmp_df = pd.read_csv(path_to_csv, encoding='cp1252')
        #print("********",tmp_df)#entire csv file
        self.transform = tranforms

        self.path_to_ads = tmp_df['Image_loc']
        #print("self.path_to_ads***",self.path_to_ads)#the 
        self.labels = tmp_df['Keywords']


        print(f'Loaded {self.labels.shape[0]} ads')



    def __getitem__(self, index):
        #how to get a single item
        #What is needed is music vgg features extracted, artist name and genre

        img = Image.open(self.path_to_ads[index])
        #img = make_square(img)
        img = img.convert('RGB')
        #img.save('result.png')
        if self.transform is not None:
            img = self.transform(img)


        label = (self.labels[index])
        return img, label

    def __len__(self):
        return self.path_to_ads.shape[0]


class AdvertDataset_batch(Dataset):
    """An override of class Dataset.

    All  datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __init__(self,  path_to_csv, tranforms= None):
        #some arguments needed for music dataset to function?
        #place init of them here, like csv, folder_loc, etc
        tmp_df = pd.read_csv(path_to_csv, encoding='cp1252')
        self.transform = tranforms

        self.path_to_ads = tmp_df['Image_loc']
        self.labels = tmp_df['Keywords']


        print(f'Loaded {self.labels.shape[0]} ads')



    def __getitem__(self, index):
        #how to get a single item
        #What is needed is music vgg features extracted, artist name and genre

        img = Image.open(self.path_to_ads[index])
        img = make_square(img)
        #img = img.convert('RGB')
        #img.save('result.png')
        if self.transform is not None:
            img = self.transform(img)

        label = (self.labels[index])
        return img, label

    def __len__(self):
        return self.path_to_ads.shape[0]

class AdvertDataset_batch_test(Dataset):
    """An override of class Dataset.

    All  datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __init__(self,  path_to_csv, tranforms= None):
        #some arguments needed for music dataset to function?
        #place init of them here, like csv, folder_loc, etc
        tmp_df = pd.read_csv(path_to_csv, encoding='cp1252')
        self.transform = tranforms

        self.path_to_ads = tmp_df['image_loc']
        self.labels = tmp_df['keywords']


        print(f'Loaded {self.labels.shape[0]} ads')



    def __getitem__(self, index):
        #how to get a single item
        #What is needed is music vgg features extracted, artist name and genre
        
        img = Image.open(self.path_to_ads[index])#......
        #img.replace('\\', '/')
        img = make_square(img)
        #img = img.convert('RGB')
        #img.save('result.png')
        if self.transform is not None:
            img = self.transform(img)

        label = (self.labels[index])
        return img, label

    def __len__(self):
        return self.path_to_ads.shape[0]


#dataset = AdvertDataset(r'data/gt.csv', test_transform)

#print(dataset.__getitem__(1),(dataset.path_to_ads[1]))

