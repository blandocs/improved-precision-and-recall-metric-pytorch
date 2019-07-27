import os, torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset  # For custom datasets

from torchsummary import summary
from tqdm import tqdm

# https://pytorch.org/tutorials/advanced/neural_style_tutorial.html

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.device_count(), "GPUS!")

class feature_extractor(object):
    def __init__(self, args):
        # parameters
        self.args = args
        self.generated_dir = args.generated_dir
        self.real_dir = args.real_dir
        self.batch_size = args.batch_size
        self.cpu = args.cpu
        self.data_size = args.data_size

    def extract(self):
        # test loading image properly
        # self.show_image(img)

        cnn = models.vgg16(pretrained=True)

        # https://discuss.pytorch.org/t/how-to-delete-layer-in-pretrained-model/17648/5
        # extract 2nd FC ReLU

        # 아래 말고 다음과 같이 뽑아 낼 수 도 있음. content_targets = [A.detach() for A in vgg(content_image, content_layers)] 
        # 다음 URL 참조. https://github.com/leongatys/PytorchNeuralStyleTransfer
        cnn.classifier = nn.Sequential(*[cnn.classifier[i] for i in range(5)])
        cnn = cnn.to(device).eval()
        # summary(cnn, (3, 224, 224))

        generated_features = []
        real_features = []
        generated_img_paths = []

        with torch.no_grad():

            generated_data = ImageDataset(self.generated_dir, self.data_size, self.batch_size)
            generated_loader = DataLoader(generated_data, batch_size=self.batch_size, shuffle=False)

            for imgs, img_paths in tqdm(generated_loader, ncols=80):
                target_features = cnn(imgs)

                img_paths = list(img_paths)
                generated_img_paths.extend(img_paths)

                for target_feature in torch.chunk(target_features, target_features.size(0), dim=0):
                    generated_features.append(target_feature)

            real_data = ImageDataset(self.real_dir, self.data_size, self.batch_size)
            real_loader = DataLoader(real_data, batch_size=self.batch_size, shuffle=False)

            for imgs, _ in tqdm(real_loader, ncols=80):
                target_features = cnn(imgs)

                for target_feature in torch.chunk(target_features, target_features.size(0), dim=0):
                    real_features.append(target_feature)

        return generated_features, real_features, generated_img_paths

    def show_image(self, img):
        unloader = transforms.ToPILImage()  # reconvert into PIL image
        plt.ion()
        plt.figure()
        image = img.cpu().clone()  # we clone the tensor to not do changes on it
        image = image.squeeze(0)      # remove the fake batch dimension
        image = unloader(image)
        plt.imshow(image)
        plt.title(Image)
        plt.pause(10) # pause a bit so that plots are updated


class ImageDataset(Dataset):
    def __init__(self, dir_path, data_size=100, batch_size=64):
        self.dir_path = dir_path

        data_size = data_size - data_size%batch_size

        self.img_paths = []

        for i, img_name in enumerate(os.listdir(dir_path)):
            if i >= data_size:
                break
            img_path = os.path.join(dir_path, img_name)
            self.img_paths.append(img_path)

        self.imsize = 224 # for vgg input size

        # https://github.com/leongatys/PytorchNeuralStyleTransfer
        self.transformations = transforms.Compose([
            transforms.Resize(self.imsize),  # scale imported image
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to BGR
            transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], #subtract imagenet mean
            std=[1,1,1]),
            transforms.Lambda(lambda x: x.mul_(255)),
            ])  # transform it into a torch tensor

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path)
        image = self.transformations(image)
        return image.to(device, torch.float), img_path

    def __len__(self):
        return len(self.img_paths)