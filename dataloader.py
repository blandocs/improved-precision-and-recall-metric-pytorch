import os, torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models

from torchsummary import summary
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
        self.seed = args.seed

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
        
        # extract generated images
        for img_name in os.listdir(self.generated_dir):
            img_name = os.path.join(self.generated_dir, img_name)
            img = self.image_loader(img_name)
            # print(img)
            target_feature = cnn(img)
            # print(target_feature)

        # real generated images
        for img_name in os.listdir(self.real_dir):
            img_name = os.path.join(self.real_dir, img_name)
            img = self.image_loader(img_name)
            # print(img)
            target_feature = cnn(img)
            # print(target_feature)


    def image_loader(self, image_name):
        image = Image.open(image_name)
        # print(image)
        # fake batch dimension required to fit network's input dimensions
        image = self.transformations(image).unsqueeze(0)
        return image.to(device, torch.float)

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


