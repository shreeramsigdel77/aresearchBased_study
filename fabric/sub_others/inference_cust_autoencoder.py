import torch
from torch import tensor
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, dataloader
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms.transforms import Resize
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import os
import cv2
import numpy as np

from PIL import Image
from natsort import natsorted
from torch.utils.data import dataset

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

torch.backends.cudnn.benchmark = True


#write on tensorboard 
from torch.utils.tensorboard import SummaryWriter
tens_writer = SummaryWriter(log_dir='log_output')

#Writer will output to ./runs/ directory by default.

if not os.path.exists('./cust_dc_img'):
    os.mkdir('./cust_dc_img')



import matplotlib.pyplot as plt

# def load_ckp(checkpoint_fpath, model, optimizer):
#     checkpoint = torch.load(checkpoint_fpath)
#     # state_dict = torch.load('./cust_conv_autoencoder.pth')
#     # model.load_state_dict(state_dict)
#     model.load_state_dict(checkpoint['state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer'])
#     return model, optimizer, checkpoint['epoch']

def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    # state_dict = torch.load('./cust_conv_autoencoder.pth')
    # model.load_state_dict(state_dict)
    model.load_state_dict(checkpoint['state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
    return model


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def create_dir(dir_name):
    os.makedirs(dir_name) if not os.path.exists(dir_name) else print("Exist")

def preview_img(pic, dir_path):
    create_dir(dir_path)
    for count,i in enumerate (pic):
        # print(i)
        i = i.T
        # print(i.shape)
        # print(type(i))
        # npimg = img.numpy()
        npimg = i
        i = i / 2 + 0.5     # unnormalize

        # cv2.imshow("preview", i)
        # cv2.waitKey(1000)
        # cv2.imwrite(os.path.join(dir_path,f"{count}.png"),i,cv2.CV_16U)

        # from PIL import Image
        # img = np.array(Image.fromarray((i * 255).astype(np.uint8)).resize((400, 400)))
        img = np.array(Image.fromarray((i * 255).astype(np.uint8)).resize((400, 400)).convert('RGB'))
        im = Image.fromarray(img)
        #flip vertically
        im = im.transpose(Image.ROTATE_270)
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
        im.save(os.path.join(dir_path,f"{count}.png"))


        # plt.figure(figsize=(400,400))
        # plt.imshow(i)
        # # plt.show()
        # plt.savefig(os.path.join(dir_path,f"{count}.png"))
        # exit()












def image_np_array(image_dir):
    """[Converts list of image from a directory to numpy array]

    Args:
        image_dir ([string]): [image directory path]

    Returns:
        [numpy.ndarray]: [list of images from a folder]
    """
    img_list = []
    for i in (os.listdir(image_dir)):
        img = cv2.imread(os.path.join(image_dir,i))
        # print(type(img))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        cv2.imshow("window", img)
        cv2.waitKey(1000)
        # print(img.shape)
        img_list.append(img)
    #convert list to numpy array
    img_list = np.asarray(img_list)
   
    return img_list

class CustomDataSet(dataset.Dataset):
    def __init__(self,main_dir,transform) -> None:
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        self.total_imgs = natsorted(all_imgs)
    
    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self,idx):
        img_loc =os.path.join(self.main_dir,self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 400, 400)
    return x


train_dir = "/home/pasonatech/workspace/deepNN/aresearchBased_study/fabric/data/train"
test_dir = "/home/pasonatech/workspace/deepNN/aresearchBased_study/fabric/data/val"
inference_dir = "/home/pasonatech/workspace/deepNN/aresearchBased_study/fabric/data/resize_output"

train_output = "/home/pasonatech/workspace/deepNN_py/pytorch_autoencoder/cust_dc_img"
# results_dir = "/home/pasonatech/workspace/deepNN/aresearchBased_study/fabric/inference_results"
# image_np_array(inference_dir)



num_epochs = 3000
# batch_size = 128
learning_rate = 1e-3

# img_transform = transforms.Compose([
#     transforms.Resize((400,400)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
# ])



img_transform = transforms.Compose([
    transforms.Resize((400,400)),
    transforms.RandomHorizontalFlip(p=0),
    transforms.RandomVerticalFlip(p=0),
    transforms.RandomRotation(0),
    transforms.ToTensor(),
    transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ], std = [ 0.229, 0.224, 0.225 ]),
    
    
])




# dataset = MNIST('./data', transform=img_transform,download= True)
# print(dataset[0])
# print(len(dataset[0]))
# print(type(dataset[0]))
# exit()

#calling my dataset

#parameters 
params = {
    'batch_size' : 32,
    'shuffle': True,
    'num_workers': 6
}


mytrain_dataset = CustomDataSet(main_dir=train_dir,transform=img_transform)
# train_loader = DataLoader(my_dataset,**params)
mytest_dataset = CustomDataSet(main_dir=inference_dir,transform=img_transform)

dataloader = DataLoader(mytrain_dataset,**params)

testdataloader = DataLoader(mytest_dataset, **params)


# get some random training images
dataiter = iter(dataloader)
images = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images,))



# dataloader = DataLoader(dataset, **params)

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(                                                   # 3,  400, 400
            nn.Conv2d(in_channels = 3, out_channels= 8, kernel_size = 3, padding=1),   #out 8 400,400
            nn.ReLU(True),
            nn.Conv2d(in_channels = 8, out_channels= 16, kernel_size = 3, padding=1),  # in 16 
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size = 2, stride=2),  # b, 32, 200, 200
           


            #dense layer
        )
        self.decoder = nn.Sequential(
            #dense layer

            nn.ConvTranspose2d(16, 8, 2, stride=2),  # b , 64, 
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 3, 3, stride=1,padding=1),    
            nn.Tanh()

        )


    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


model = autoencoder().cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             weight_decay=1e-5)


#load trained model


ckp_path = "/home/pasonatech/workspace/deepNN_py/pytorch_autoencoder/model_dir/checkpoint_2999.pth"

if os.path.exists(ckp_path):
    model, optimizer1, start_epoch1 = load_ckp(ckp_path, model, optimizer)



for data in testdataloader:
    # print(data)
    # img, _ = data
    img= data
    # print(type(img))
    
    img = Variable(img).cuda()
    # ===================forward=====================
    output = model(img)
    # loss = criterion(output, img)
    # ===================backward====================
    #write loss function on tensorboard
        
    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()
# ===================log========================
    
# pic = to_img(output.cpu().data)
pic = output.cpu().detach().numpy()
preview_img(pic,os.path.join(train_output,"inference"))
print(type(pic))
print(pic.shape)





# total_loss = 0
# for epoch in range(num_epochs):
#     for data in dataloader:
#         # print(data)
#         # img, _ = data
#         img= data
#         # print(type(img))
        
#         img = Variable(img).cuda()
#         # ===================forward=====================
#         output = model(img)
#         loss = criterion(output, img)
#         # ===================backward====================
#         #write loss function on tensorboard
#         tens_writer.add_scalar("Loss/train", loss, epoch)
        
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     # ===================log========================
#     total_loss += loss.data
#     print('epoch [{}/{}], loss:{:.4f}'
#           .format(epoch+1, num_epochs, total_loss))
#     if epoch % 10 == 0:
       
#         # pic = to_img(output.cpu().data)
#         pic = output.cpu().detach().numpy()
#         preview_img(pic,os.path.join(train_output,str(epoch)))
#         print(type(pic))
#         print(pic.shape)
        

#flush make sure all the pending events have been written to disk
# tens_writer.flush()

# torch.save(model.state_dict(), './cust_conv_autoencoder.pth')