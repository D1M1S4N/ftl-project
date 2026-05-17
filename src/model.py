import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

def get_resnet18_2d(in_channels=1, num_classes=2):
    # Cargamos el modelo base (se puede usar pesos preentrenados si queremos transfer learning previo)
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    
    # 1. Adaptar la primera capa convolucional para que acepte 1 canal en lugar de 3
    if in_channels != 3:
        original_conv1 = model.conv1
        model.conv1 = nn.Conv2d(
            in_channels, 
            original_conv1.out_channels, 
            kernel_size=original_conv1.kernel_size, 
            stride=original_conv1.stride, 
            padding=original_conv1.padding, 
            bias=False
        )
        # Opcional: inicializar los pesos de la nueva capa promediando los pesos RGB originales
        with torch.no_grad():
            model.conv1.weight[:] = original_conv1.weight.mean(dim=1, keepdim=True)

    # 2. Adaptar la capa final (Fully Connected) para 2 clases (EM vs Control)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model