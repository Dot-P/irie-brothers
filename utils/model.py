import torch
import torch.nn as nn
import torchvision.models as models

def get_model(model_name: str, num_classes: int):
    """
    指定されたモデルを取得し、出力層を指定したクラス数に変更する。

    Args:
        model_name (str): モデル名
        num_classes (int): 出力クラス数

    Returns:
        nn.Module: 指定したモデル
    """
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=num_classes)
        print(model)
    elif model_name == 'alexnet':
        model = models.alexnet(pretrained=True)
        model.classifier[6] = nn.Linear(in_features=model.classifier[6].in_features, out_features=num_classes)
        print(model)
    elif model_name == 'convnext': # 良
        model = models.convnext_tiny(pretrained=True)
        model.classifier[2] = nn.Linear(in_features=model.classifier[2].in_features, out_features=num_classes)
        print(model)
    elif model_name == 'efficientnet': # 良
        model = models.efficientnet_b4(pretrained=True)
        model.classifier[1] = nn.Linear(in_features=model.classifier[1].in_features, out_features=num_classes)
        print(model)
    elif model_name == 'efficientnet_v2':
        model = models.efficientnet_v2_s(pretrained=True)
        model.classifier[1] = nn.Linear(in_features=model.classifier[1].in_features, out_features=num_classes)
        print(model)
    elif model_name == 'googlenet':
        model = models.googlenet(pretrained=True)
        print(model)
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=num_classes)
    elif model_name == 'resnext50': # 良
        model = models.resnext50_32x4d(pretrained=True)
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=num_classes)
        print(model)
    elif model_name == 'swin': # 優
        model = models.swin_t(pretrained=True)
        model.head = nn.Linear(in_features=model.head.in_features, out_features=num_classes)
        print(model)
    elif model_name == 'swin_v2':
        model = models.swin_v2_t(pretrained=True)
        model.head = nn.Linear(in_features=model.head.in_features, out_features=num_classes)
        print(model)
    else:
        raise ValueError(f"Unsupported model name: {model_name}.")
    
    return model
