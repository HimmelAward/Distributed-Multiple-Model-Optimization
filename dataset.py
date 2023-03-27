from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets,transforms
import configs

def get_dataloader():
    train_data = datasets.MNIST(root='data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)
    test_data = datasets.MNIST(root='data',
                           train=False,
                           transform=transforms.ToTensor(),
                           )
    train_dataloader = DataLoader(dataset=train_data,
                    batch_size=configs.BATCH_SIZE,
                        shuffle=False,
                        sampler=DistributedSampler(train_data))
    test_dataloader = DataLoader(dataset=test_data,
                       batch_size=configs.BATCH_SIZE,
                       shuffle=False,
                       sampler=DistributedSampler(test_data))
    return train_dataloader,test_dataloader

def get_dataloader2():
    train_data = datasets.MNIST(root='data',
                                train=True,
                                transform=transforms.ToTensor(),
                                download=True)
    test_data = datasets.MNIST(root='data',
                               train=False,
                               transform=transforms.ToTensor(),
                               )
    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=configs.BATCH_SIZE,
                                  shuffle=True,
                                  )
    test_dataloader = DataLoader(dataset=test_data,
                                 batch_size=configs.BATCH_SIZE,
                                 shuffle=True,
                                    )
    return train_dataloader, test_dataloader