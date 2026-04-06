import torch
from PIL import Image
from torchvision import transforms

# normalisation standard ImageNet
_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])


def preprocess_image(image):
    return _transform(image.convert("RGB")).unsqueeze(0)
