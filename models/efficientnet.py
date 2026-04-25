import torchvision.models as models
import torch.nn as nn

def get_efficientnet(num_classes=10, pretrained=False):
    model = models.efficientnet_b0(
        weights="IMAGENET1K_V1" if pretrained else None
    )
    # Replace classifier head for CIFAR-10
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features, num_classes
    )
    return model