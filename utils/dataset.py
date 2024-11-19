import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch

class MyDataset(Dataset):
    def __init__(self, X, y, valid=False):
        self.X = X  # データは[サンプル数, 高さ, 幅, チャネル数]の形式
        self.y = y

        # データ拡張や変換
        if not valid:
            self.trans = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),  # PILからテンソルに変換
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.trans = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, pos):
        image = self.X[pos]  # 元の形状を保持
        image = self.trans(image)  # PIL形式に変換してから処理
        label = torch.tensor(self.y[pos], dtype=torch.long)
        return image, label