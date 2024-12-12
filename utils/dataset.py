import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch

class MyDataset(Dataset):
    def __init__(self, X, y=None, type='train'):
        self.X = X  # データは[サンプル数, 高さ, 幅, チャネル数]の形式
        self.y = y

        # データ拡張や変換
        if type == 'train':
            self.trans = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),  # PILからテンソルに変換
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        elif type == 'val':
            self.trans = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        elif type == 'unlabeled':
            self.trans = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        elif type == 'no_trans':
            self.trans = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
        else:
            raise ValueError('type must be "train", "val", or "unlabeled".')

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        image = self.X[idx]
        image = self.trans(image)
        if self.y is not None:
            label = torch.tensor(self.y[idx], dtype=torch.long)
            return image, label
        else:
            return image
