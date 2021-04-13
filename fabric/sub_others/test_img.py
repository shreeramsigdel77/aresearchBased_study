import torch
from torch import tensor
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, dataloader
from torchvision import transforms
from torchvision.transforms.functional import pad, to_pil_image
from torchvision.transforms.transforms import RandomHorizontalFlip, RandomRotation, Resize
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import os
import cv2
import numpy as np

from PIL import Image
from natsort import natsorted
from torch.utils.data import dataset
# import shutil



#save checkpoint
def save_ckp(state, checkpoint_dir, epoch):
    f_path = os.path.join(checkpoint_dir , f'checkpoint_{epoch}.pth')
    torch.save(state, f_path)
    # if is_best:
    #     best_fpath = best_model_dir / 'best_model.pth'
    #     shutil.copyfile(f_path, best_fpath)

#load checkpoint
def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    # state_dict = torch.load('./cust_conv_autoencoder.pth')
    # model.load_state_dict(state_dict)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch']

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
        # image = Image.open(img_loc).convert("RGB")
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        
        # print("Getitem",type(tensor_image))       
        # img = tensor_image
        # img = img / 2 + 0.5    # unnormalize
        # print(type(img))
        # npimg = img.numpy()
        # print("preview with gett method")
        # plt.imshow(np.transpose(npimg, (1, 2, 0)))
        # plt.show()


        return tensor_image




def to_img(x, inv_normalize):
    # x = inv_normalize(x)

    from torchvision.utils import save_image
    

    # x = x.clamp("RGB")
    # x = x.view(x.size(0), 3, 400, 400)
    save_image(inv_normalize(x[0]),"img1.png")
    return x




train_dir = "/home/pasonatech/workspace/deepNN/aresearchBased_study/fabric/data/train"
test_dir = "/home/pasonatech/workspace/deepNN/aresearchBased_study/fabric/data/val"
inference_dir = "/home/pasonatech/workspace/deepNN/aresearchBased_study/fabric/data/resize_output"

train_output = "/home/pasonatech/workspace/deepNN_py/pytorch_autoencoder/cust_dc_img"
model_dir = os.path.join("/home/pasonatech/workspace/deepNN_py/pytorch_autoencoder","model_dir")
# create_dir(model_dir)

train_dir = inference_dir

ckp_path = ""










num_epochs = 10
# batch_size = 128
learning_rate = 1e-3

img_transform = transforms.Compose([
    transforms.Resize((400,400)),
    transforms.RandomHorizontalFlip(p=0),
    transforms.RandomVerticalFlip(p=0),
    transforms.RandomRotation(0),
    transforms.ToTensor(),
    # transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])

inv_normalize = transforms.Normalize((-0.485/0.229,-0.456/0.224,-0.406/0.225), (1/0.229,1/0.224,1/0.225))

#calling my dataset

#parameters 
params = {
    'batch_size' : 32,
    'shuffle': True,
    'num_workers': 6
}


my_dataset = CustomDataSet(main_dir=train_dir,transform=img_transform)
dataloader = DataLoader(my_dataset,**params)

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 3, 2, stride=2, padding=1),  # b, 1, 28, 28
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
total_loss = 0
for epoch in range(num_epochs):
    for data in dataloader:
        img = data
        img = Variable(img).cuda()
        # ===================forward=====================
        output = model(img)
        loss = criterion(output, img)
        # ===================backward====================
        #write loss function on tensorboard
        tens_writer.add_scalar("Loss/train", loss, epoch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    total_loss += loss.data
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch+1, num_epochs, total_loss))
    if epoch % 10 == 0:
        pic = to_img(output.cpu().data , inv_normalize)
        save_image(pic, './dc_img/image_{}.png'.format(epoch))

#flush make sure all the pending events have been written to disk
tens_writer.flush()

torch.save(model.state_dict(), './conv_autoencoder.pth')