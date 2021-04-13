import torch
from torch import tensor
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, dataloader
from torchvision import transforms
from torchvision.transforms.functional import pad, to_pil_image
from torchvision.transforms.transforms import RandomHorizontalFlip, RandomRotation, Resize, ToPILImage
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import os
import cv2
import numpy as np
import albumentations as A


from PIL import Image
from natsort import natsorted
from torch.utils.data import dataset
# import shutil



inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.255]
)
# inv_tensor = inv_normalize(tensor)











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



import matplotlib.pyplot as plt


def cust_imshow(img):
    #torch.tensor data
    print(img[0])
    img = img / 2 + 0.5    # unnormalize
    print(type(img))
    npimg = img.numpy()   #converts images to numpy
    print("preview with imshow methode")
    plt.imshow(np.transpose(npimg, (1, 2, 0))) #convert from tensor image
    plt.show()

def create_dir(dir_name):
    os.makedirs(dir_name) if not os.path.exists(dir_name) else print("Directory Exist...\n Continue Trainning..")

def preview_img(pic, dir_path):
    create_dir(dir_path)
    for count,i in enumerate (pic):
        # print(i)
        print(i.shape)

        #new approach
        untransform = transforms.Compose([
            transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ], std = [ 0.229, 0.224, 0.225 ]),
            transforms.ToPILImage,
        ])
        # print(type(i))
        # cust_imshow(i)
        im_numpy = inv_normalize(i)
        # im_numpy = transforms.transforms(im_numpy)
        # print(im_numpy.shape)
        im_ram = np.transpose(im_numpy,(1,2,0))

        im_ram = im_ram.numpy()
        # plt.imshow(im_ram)
        # plt.show()
        # im_ram = untransform(i[0])
        # print("im_ram_type",type(im_ram))
        
        
        plt.imshow(im_ram)
        
        plt.tight_layout(pad=0,h_pad=0,w_pad=0,rect=(0,0,0,0))
        # plt.axes().get_xaxis().set_visible(False)
        # plt.axes().get_yaxis().set_visible(False)
        plt.axis('off')
        
        plt.savefig(os.path.join(dir_path,f"{count}.png"),pad_inches = 0)
        # im_ram.save(os.path.join(dir_path,f"{count}.png"))
        # plt.show()


        #size problem while saving
        #lets crop and adjust it
        # img_load = cv2.imread(os.path.join(dir_path,f"{count}.png")) 
        
        # resiz_img = cv2.resize(img_load, (400,400), interpolation = cv2.INTER_AREA)
        # cv2.imwrite(os.path.join(dir_path,f"{count}.png"),resiz_img)



        # im.show()

        # im_ram.show()
        # exit()


        
        # print(type(i))
        # npimg = img.numpy()
        
        # exit()
        # cv2.imshow("preview", i)
        # cv2.waitKey(1000)
        # cv2.imwrite(os.path.join(dir_path,f"{count}.png"),i,cv2.CV_16U)
        # exit()
        # from PIL import Image
        # img = np.array(Image.fromarray((i * 255).astype(np.uint8)).resize((400, 400)))
        

        #commented for test
        # i = i.T
        # print(i.shape)
        # npimg = i
        # i = i / 2 + 0.5     # unnormalize
        # img = np.array(Image.fromarray((i * 255).astype(np.uint8)).resize((400, 400)).convert('RGB'))
        # im = Image.fromarray(img)
        # #flip vertically
        # im = im.transpose(Image.ROTATE_270)
        # im = im.transpose(Image.FLIP_LEFT_RIGHT)
        # im.save(os.path.join(dir_path,f"{count}.png"))
        # im.show()

        # plt.figure(figsize=(400,400))
        # plt.imshow(i)
       
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
        # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        # cv2.imshow("window", img)
        # cv2.waitKey(1000)
        # print(img.shape)
        img_list.append(img)
    #convert list to numpy array
    img_list = np.asarray(img_list)
   
    return img_list
#  print(i.shape)
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

# def to_img(x):
#     x = 0.5 * (x + 1)
#     # x = x.clamp(0, 1)
#     x = x.view(x.size(0), 3, 400, 400)
#     return x



def to_img(x):
    cust_imshow(x[0])
    x = 0.5 * (x + 1)
    # x = x / 2 + 0.5  #unnormalize
   
    print(x[0])
    print(type(x[0]))
    print((x[0].shape))
    # x = x.clamp(0, 1)
    from torchvision.utils import save_image
    save_image(x[0],"img1.png")
    exit()
    x = x.view(x.size(0), 3, 400, 400)
    return x


train_dir = "/home/pasonatech/workspace/deepNN/aresearchBased_study/fabric/data/train"
test_dir = "/home/pasonatech/workspace/deepNN/aresearchBased_study/fabric/data/val"
inference_dir = "/home/pasonatech/workspace/deepNN/aresearchBased_study/fabric/data/resize_output"

train_output = "/home/pasonatech/workspace/deepNN_py/pytorch_autoencoder/cust_dc_img"
model_dir = os.path.join("/home/pasonatech/workspace/deepNN_py/pytorch_autoencoder","model_dir")
create_dir(model_dir)

# train_dir = inference_dir

ckp_path = "/home/pasonatech/workspace/deepNN_py/pytorch_autoencoder/model_dir/checkpoint_2999.pth"

# results_dir = "/home/pasonatech/workspace/deepNN/aresearchBased_study/fabric/inference_results"
# image_np_array(inference_dir)



num_epochs = 500
# batch_size = 128
learning_rate = 1e-3

img_transform = transforms.Compose([
    transforms.Resize((400,400)),
    transforms.RandomHorizontalFlip(p=0),
    transforms.RandomVerticalFlip(p=0),
    transforms.RandomRotation(0),
    transforms.ToTensor(),
    transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ], std = [ 0.229, 0.224, 0.225 ]),
    
    
])

# albu_transform = A.Compose([
    
# ])

# (0.5,0.5,0.5), (0.5,0.5,0.5)
# mean = [ 0.485, 0.456, 0.406 ], std = [ 0.229, 0.224, 0.225 ]


#calling my dataset

#parameters 
params = {
    'batch_size' : 32,
    'shuffle': True,
    'num_workers': 6
}


my_dataset = CustomDataSet(main_dir=train_dir,transform=img_transform)







# train_loader = DataLoader(my_dataset,**params)

dataloader = DataLoader(my_dataset,**params)



# get some random training images
dataiter = iter(dataloader)
images = dataiter.next()

# show images
# cust_imshow(torchvision.utils.make_grid(images,))
# cust_imshow(images[0])


# dataloader = DataLoader(dataset, **params)

#output size
    
#output_size = 1+(input_size-kernel_size+2*padding_size)/stride



class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(                                                   # 3,  400, 400
            nn.Conv2d(in_channels = 3, out_channels= 8, kernel_size = 3, padding=1),   #out 8 400,400
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size = 2, stride=1,padding=1),
            nn.Conv2d(in_channels = 8, out_channels= 16, kernel_size = 3, padding=1),  # in 16 
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size = 2, stride=2),  
           


            #dense layer
        )
        self.decoder = nn.Sequential(
            #dense layer

            nn.ConvTranspose2d(16, 8, 2, stride=2),  # 16 8
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 3, 3, stride=1,padding=1),   # 8 3 
            nn.Tanh()

        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x






start_epoch=0
model = autoencoder().cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             weight_decay=1e-5)
if os.path.exists(ckp_path):
    model, optimizer, start_epoch = load_ckp(ckp_path, model, optimizer)

total_loss = 0
for epoch in range(start_epoch,num_epochs):
    for data in dataloader:
        # print(data)
        # img, _ = data
        img= data
        # print("epoch img type",type(img))
        
        #noise
        noise = np.random.normal(0, .1, img.shape)
        new_signal = img + noise

        # new_img = Variable(new_signal).cuda()
        img = Variable(img).cuda()
        # ===================forward=====================
        output = model(img)
        # loss = criterion(output.cuda(), img.cuda())
        loss = criterion(output.cuda(), img.cuda())
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
    if (epoch+1) % 100 == 0:  
        
        # pic = to_img(output.cpu().data)
        # save_image(pic, './dc_img/image_{}.png'.format(epoch))        
        
        # pic = output.cpu().detach().numpy()
        pic = output.data.cpu().detach()
        print("pic type",type(pic))





        preview_img(pic,os.path.join(train_output,str(epoch)))
        
    # if (epoch+1) % 5 == 0:
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        save_ckp(checkpoint, model_dir,epoch)
    if epoch+1 == num_epochs:
        save_ckp(checkpoint, model_dir,"final")

#flush make sure all the pending events have been written to disk
tens_writer.flush()
# torch.save(model.state_dict(), os.path.join(model_dir,'final_model.pth'))





