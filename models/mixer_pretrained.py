import timm
import torch.nn as nn

def get_pretrained_mixer(num_classes=10):
    # Mixer-B/16 pretrained on ImageNet-21k
    model = timm.create_model(
        "mixer_b16_224_miil_in21k",
        pretrained=True
    )
    # Replace classifier head for our dataset
    model.head = nn.Linear(model.head.in_features, num_classes)
    return model