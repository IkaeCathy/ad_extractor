'''
Traning Code for pycharm input transformers.
'''

from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
from configargparse import YAMLConfigFileParser
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
#from yaap import ArgParser***********
from argparse import ArgumentParser
from tessocrWrapper import read_keys_from_ad
from dataset_loaders import AdvertDataset
from tessocr import read_keys


class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.sig2 = nn.Sigmoid()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2),  # 24x24x8
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 5, 1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 8, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 3, 3, 1, padding=1),  # 40x395x395

        )

    def encode(self, x: Variable) -> (Variable, Variable):
        return self.sig2(self.encoder(x))

    def forward(self, x: Variable) -> (Variable, Variable):
        out = self.encode(x)
        # out = F.adaptive_avg_pool2d(out, (1200, 1200))
        return out


model = AE()
if torch.cuda.is_available():
    model = model.cuda()
model.load_state_dict(torch.load(r'models\auto_encoder_model.pt'))
print('model loaded')

def get_accuracy(tesser_output, label):
    labels_list = list(filter(None, label))
    # flatten
    # labels_list = [item for sublist in labels_list for item in sublist]
    tesser_list_output = list(filter(None, tesser_output))
    # make into seperate strng words
    labels_list = labels_list[0][2:-2].replace("'", "").split(',')
    #labels_list = ','.join(labels_list).replace("'", "").split(',')
    tesser_list_output = ','.join(tesser_list_output).split(',')
    # transform to counter structure
    labels_list = Counter(labels_list)
    tesser_list_output = Counter(tesser_list_output)
    a = Variable(torch.from_numpy(np.asarray(sum(labels_list.values()), dtype=np.float32))).view(1,
                                                                                                 -1)  # number of words
    b = Variable(torch.from_numpy(np.asarray(sum((labels_list - tesser_list_output).values()), dtype=np.float32))).view(
        1, -1)  # the missing words
    tess_seen = (a - b)/a
    return tess_seen*100


def plot_image(tensor):
    plt.figure()
    # imshow needs a numpy array with the channel dimension
    # as the the last dimension so we have to transpose things.
    plt.imshow(tensor.detach().cpu().squeeze(0).numpy().transpose(1, 2, 0))
    plt.show()


def plot_numpy_image(numpy):
    plt.figure()
    # imshow needs a numpy array with the channel dimension
    # as the the last dimension so we have to transpose things.
    plt.imshow(numpy, cmap='gray')
    plt.show()


def plot_grayscale_tensor(tensor):
    plt.figure()
    plt.imshow(tensor.detach().cpu().squeeze(0).numpy(), cmap='gray')
    plt.show()



def test(epoch):
    model.eval()
    test_accu = 0
    for batch_idx, data in enumerate(test_loader):
        image, keys = data
        with torch.no_grad():
            #out = model(image.cuda())
            out = model(image.cpu())
            #out= image*255
            nd_out = out.clone()
            nd_out = nd_out.detach().cpu().squeeze(0).numpy().transpose(1, 2, 0)
            # plot_grayscale_image(image.squeeze(0))
            # plot_numpy_image(nd_out)
            #tesser_out = read_keys_from_ad(nd_out)
            #tesser_out = read_keys_from_ad(nd_out)
            tesser_out = read_keys(nd_out)
            accu = get_accuracy(tesser_out, keys)
            #print(f'Image accuracy:{accu.data.item()}% \t, keywords= {tesser_out}')
            test_accu += accu
    test_accu /= len(test_loader)
    print('====> Data set accuracy: {:.4f}'.format(test_accu.data.item()))
    return test_accu

def parse_args(): # add this ot the train of model in R&D
    #parser = ArgParser(allow_config=True,
                       #config_file_parser_class=YAMLConfigFileParser)
    parser = ArgumentParser()
    parser.add("--csv-file", type=str, required=True, default=r"data\gt.csv",
               help="Path to csv file that contains "
                    "Image location and the corresponding list keywords "
               )
    parser.add("--model-name", type=str, required=True, default=r"data\gt.csv",
               help="name of model to save. Specify different names for different models when training to save "
                    "multiple models. At each run, current model name is loaded "
               )
    parser.add("--epochs", type=int, required=True, default=5,
               help="Number of times to run cycle through training data "
               )

    parser.add("--log-interval", type=int, required=True, default=10,
               help="Period to display training progress, corresponding to the number of samples seen "
               )
    parser.add("--source-folder", type=str, default=r"data\nondata",
               help="Path to directory that contains "
                    "a set of image files where or sub-directory of images "
               )
    parser.add("--destination", type=str, default=r"data\output",
               help="Path to a folder to save extracted .txt files.")
    args = parser.parse_args()
    return args


def main():

    for epoch in range(1, EPOCHS + 1):
        # train_loss, train_accu = train(epoch)
        _ = test(epoch)



if __name__ == '__main__':
    args = parse_args()

    CUDA = torch.cuda.is_available()
    BATCH_SIZE = 1
    EPOCHS = 1
    kwargs = {'num_workers': 0, 'pin_memory': True} if CUDA else {}
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    # Load dataser from CSV with traininig information
    ads_dataset = AdvertDataset(args.csv_file,
                                test_transform)  # if args are needed
    test_loader = DataLoader(ads_dataset, batch_size=BATCH_SIZE, **kwargs)
    main()
