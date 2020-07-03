'''
Traning Code for pycharm input transformers.
'''
##import sys
##sys.path.append("/anaconda3/lib/python3.7/site-packages")
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
from configargparse import YAMLConfigFileParser
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
#from yaap
#import argparse #ArgParser 
from argparse import ArgumentParser
from dataset_loaders import AdvertDataset
from tessocr import read_keys


class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2),  # 24x24x8
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 5, 1,padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 8, 3, 1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 3, 3, 1, padding= 1),  # 40x395x395

        )

    def encode(self, x : Variable) ->(Variable, Variable):

        return self.sig(self.encoder(x))


    def forward(self, x: Variable) -> (Variable, Variable):
        out = self.encode(x)

        #out = F.adaptive_avg_pool2d(out, (1200, 1200))
        return out


def parse_args(): # add this ot the train of model in R&D
    #parser = ArgumentParser(allow_config=True,
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


args = parse_args()
print("args.destination====",args.destination)
model = AE()
if torch.cuda.is_available():
    model = model.cuda()
else:
    map_location='cpu'
#model.load_state_dict(torch.load(r'models/%s_model.pt'%(args.model_name)), map_location='cpu')#torch.load
torch.load(r'%s_model.pt'%(args.model_name), map_location='cpu')
print('model loaded')

def get_accuracy(tesser_output, label):
    labels_list = list(filter(None, label))
    
    
    # flatten
    # labels_list = [item for sublist in labels_list for item in sublist]
    tesser_list_output = list(filter(None, tesser_output))
    # make into seperate strng words
    labels_list = labels_list[0][2:-2].replace("'", "").split(',')
    #print("labels_list =", labels_list)#actual keys from the file
    tesser_list_output = ','.join(tesser_list_output).split(',')
    #print("tesser_list_output =", tesser_list_output)
    # transform to counter structure
    labels_list = Counter(labels_list)
    tesser_list_output = Counter(tesser_list_output)
    a = Variable(torch.from_numpy(np.asarray(sum(labels_list.values()), dtype=np.float32))).view(1,
                                                                                                 -1)  # number of words
    b = Variable(torch.from_numpy(np.asarray(sum((labels_list - tesser_list_output).values()), dtype=np.float32))).view(
        1, -1)  # the missing words
    reward = Variable(torch.from_numpy(np.asarray(sum(tesser_list_output.values()), dtype=np.float32))).view(
        1, -1)
    iou = (a - b) / (a+reward)
    return iou , (a - b) / a


def loss_function(out, image, tesser_output, label):

    BCE = F.binary_cross_entropy(out, image)
    # BCE = BCE.data.sum()
    # remove empty strings
    iou, tess_seen = get_accuracy(tesser_output, label)

    return BCE-10*iou.cpu() , tess_seen #********change this to cpu
 
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
    plt.imshow(tensor.detach().cpu().squeeze(0).numpy(), cmap = 'gray')
    plt.show()

def train(epoch):
    
    model.train()
    try:
        train_loss, train_accu = 0, 0
        
        for batch_idx, data in enumerate(train_loader):
                
                image, keys = data
                #plot_grayscale_image(image.squeeze(0))
                optimizer.zero_grad()
                #out = model(image.cuda())
                out = model(image.cpu())
                #image = model.sig(image).cuda()  # put sigmoid after relus
                image = model.sig(image).cpu()  # put sigmoid after relus
                nd_out = out.clone()
                nd_out = nd_out.detach().cpu().squeeze(0).numpy().transpose(1, 2, 0)
                #plot_grayscale_image(image.squeeze(0))
                #plot_numpy_image(nd_out)
                tesser_out = read_keys(nd_out)
                #print(tesser_out)
                # if len(tesser_out[0].split(',')) < 2:
                #     plot_numpy_image(nd_out)
                loss_out, accu = loss_function(out, image,tesser_out, keys )
                loss_out.backward()
                train_loss += loss_out.data.item()
                train_accu += accu.data.item()*100
                optimizer.step()
    except:
        pass

        if batch_idx % LOG_INTERVAL == 0:# could batches is an empty list
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\toutLoss: {:.6f}'.format(
                        epoch,
                        batch_idx , len(train_loader),
                        100*batch_idx / len(train_loader),
                        loss_out.data.item() / len(train_loader)))

    print('====> Epoch: {} Average accuracy: {:.4f}'.format(
          epoch, train_accu / len(train_loader) ))
    return train_loss, train_accu / len(train_loader)

def test(epoch):
    model.eval()
    test_accu = 0
    for batch_idx, data in enumerate(test_loader):

        image, keys = data
        with torch.no_grad():

            #out = model(image.cuda())
            out = model(image.cpu())
            #image = model.sig(image).cuda()
            #image = model.sig(image).cpu()
            nd_out = out.clone()
            nd_out = nd_out.detach().cpu().squeeze(0).numpy().transpose(1, 2, 0)
            #plot_grayscale_image(image.squeeze(0))
            #plot_numpy_image(nd_out)
            tesser_out = read_keys(nd_out)
            #print("tesser_out =",tesser_out)#empty
            #print(tesser_out)
            # if len(tesser_out[0].split(',')) < 2:
            #     plot_numpy_image(nd_out)
            _,test_out = loss_function(out, image, tesser_out, keys)
            test_accu += test_out*100

    test_accu /= len(test_loader)
    # test_relu_out_loss /= len(test_loader.dataset)
    print('====> Test set accuracy: {:.4f}'.format(test_accu.data.item()))
    return test_accu

def main():
    writer = SummaryWriter()
    tmp = 0
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_accu = train(epoch)
        test_accu = test(epoch)

        #data grouping by `slash`
        writer.add_scalar('TesserOCR/Train_accu', train_accu, epoch)
        writer.add_scalar('TesserOCR/Test_accu', test_accu, epoch)

        total_accu = test_accu
        if total_accu >= tmp:
            best = total_accu
            print('saving model @', best)
            torch.save(model.state_dict(), (r'%s_model.pt'%(args.model_name)))
            tmp = best

        scheduler.step(train_loss)




# noinspection PyPackageRequirements
if __name__ == '__main__':

    CUDA = torch.cuda.is_available()
    SEED = 1
    BATCH_SIZE = 1
    LOG_INTERVAL = args.log_interval
    EPOCHS = args.epochs

    kwargs = {'num_workers': 0, 'pin_memory': True} if CUDA else {}


    test_transform = transforms.Compose([
        # transforms.Resize((600), interpolation=Image.BICUBIC),
        # transforms.ColorJitter(brightness=0.6, contrast=0.3),# saturation=0.3, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    #Load dataser from CSV with traininig information
    #csv_file="/Users/catherine/Downloads/R&D_keyword_extractorV2/R&D_keyword_extractor/Main/data/gt.csv"
    ads_dataset = AdvertDataset(args.csv_file,
                                test_transform)  # if args are needed
    # make splits
    num_train = len(ads_dataset)
    indices = list(range(num_train))
    #splits percentage for validation during training
    split = int(np.floor(0.30 * num_train))

    np.random.seed(124)
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(ads_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, **kwargs)
    test_loader = DataLoader(ads_dataset, batch_size=BATCH_SIZE, sampler=valid_sampler, **kwargs)
    #test_loader = DataLoader(ads_dataset, batch_size=BATCH_SIZE, **kwargs)
   
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)  #
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=4, mode='min',
                                  verbose=True, threshold=1)

    #run the main executing code if this is the main script to run
    main()
