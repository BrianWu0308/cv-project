import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class MyCIFAR10Dataset(Dataset):
    def __init__(self, root, train, transform=None, target_transform=None, download=True):
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.base_dataset = torchvision.datasets.CIFAR10(
            self.root, self.train, None, None, self.download
        )

    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, index):
        img, label = self.base_dataset[index]
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)
        return img, label

def build_transforms(phase: str):
    mean, std = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
    if phase == 'train':
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4), 
            transforms.RandomHorizontalFlip(), 
            transforms.ToTensor(), 
            transforms.Normalize(mean, std)
        ])
    return transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize(mean, std)
        ])

def create_dataloaders(batch_size, num_workers, root='data', download=True):
    train_transform = build_transforms('train')
    val_transform = build_transforms('val')
    train_dataset = MyCIFAR10Dataset(
        root, train=True, transform=train_transform, target_transform=None, download=download
    )
    val_dataset = MyCIFAR10Dataset(
        root, train=False, transform=val_transform, target_transform=None, download=False
    )
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_dataloader = DataLoader(
        dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_dataloader, val_dataloader