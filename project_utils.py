import torch
import glob
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from PIL import Image
from random import randint
from visualisation.core.utils import device 
from utils import *
from visualisation.core import *
from cifar_models.models_vgg import *

def load_random_image(image_class):
    cifar_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
    classes = ('plane', 'car', 'bird', 'cat',
        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    while(True):
        x = randint(0, 50000 - 1)
        image, class_index = cifar_dataset[x]
        if classes[class_index] == image_class:
            save_path = 'static/img/input_images/{}/img{}.png'.format(classes[class_index], x)
            image.save(save_path)
            return save_path
    
def load_model(model):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    models_dict = {'resnet': ResNet18(), 'vgg': VGG('VGG19'), 'efficient': EfficientNetB0()}

    net = models_dict[model]
    net = net.to(device)
    
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    
    checkpoint = torch.load('./checkpoints/{}_ckpt.pth'.format(model))
    net.load_state_dict(checkpoint['net'])
    return net

def load_layer_visualisation(image, layer, model='resnet'):
    net = load_model(model)
    input_tensor = img_to_tensor(image)
    model_traced = module2traced(net, input_tensor)
    visualised_layer = model_traced[layer]

    vis = Weights(net, device)
    images, info = vis(input_tensor, visualised_layer)
    img_list = []

    for index, tens in enumerate(images):
        img = tensor2img(tens)
        img_path = 'static/img/layer_visualisation/img{}.png'.format(index)
        plt.imsave(img_path, img)
        img_list.append(img_path)
    return img_list    

def load_saliency_map(image, layer, model='resnet'):
    net = load_model(model)
    input_tensor = img_to_tensor(image)
    model_traced = module2traced(net, input_tensor)
    visualised_layer = model_traced[layer]

    vis = SaliencyMap(net, device)
    image, info = vis(input_tensor, visualised_layer, guide=True)
    
    img = tensor2img(image)
    img_path = 'static/img/saliency_maps/img1.png'
    plt.imsave(img_path, img)
    return img_path

def img_to_tensor(image):
    transform = transforms.Compose([transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    img = Image.open(image)
    tensor = transform(img).unsqueeze(0)
    tensor = tensor.to(device)
    return tensor