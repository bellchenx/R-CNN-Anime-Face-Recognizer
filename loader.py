import os
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

image_size = 128
batch_size = 50

def get_loader(dataset):
    train_root = os.path.join(os.path.abspath(os.curdir), dataset)
    print('-- Load data from {0}.'.format(train_root))
    dataset = ImageFolder(
        root=dataset,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    return loader

def denorm(x):
    out = x * 0.5 + 0.5
    return out.clamp(0, 1)
'''
from torchvision.utils import save_image
loader = get_loader('./Data/misaka-test')
for idx, (input, target) in enumerate(loader):
    save_image(denorm(input), './test/tmp-%d.jpg' % idx)'''




