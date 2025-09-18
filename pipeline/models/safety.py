import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        
        # placeholder, will replace dynamically
        self.fc1 = None  
        self.fc2 = None
        self.num_classes = num_classes
        self.default_unsafe_threshold = 0.5

    def _get_flatten_size(self, x):
        device = next(self.parameters()).device
        x = x.to(device)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        return x.view(x.size(0), -1).shape[1]


    def build_fc(self, input_shape):
        device = next(self.parameters()).device   
        dummy = torch.zeros(1, *input_shape, device=device)
        flatten_size = self._get_flatten_size(dummy)
        self.fc1 = nn.Linear(flatten_size, 128).to(device)
        self.fc2 = nn.Linear(128, self.num_classes).to(device)

    def forward(self, x):
        if self.fc1 is None:  # build FC layers dynamically at first run
            self.build_fc(x.shape[1:])
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x





def load_safety_model(model_path: str, device: str):
    model = SimpleCNN(num_classes=2)
    checkpoint = torch.load(model_path, map_location=device)

    # Handle state_dict wrapping
    state = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    new_state = {}
    for k, v in state.items():
        new_state[k.replace("module.", "")] = v
    model.load_state_dict(new_state, strict=False)
    model.to(device).eval()
    print("✅ Safety filter model loaded")
    return model