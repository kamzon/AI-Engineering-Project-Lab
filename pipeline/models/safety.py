import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2, base_channels: int = 32):
        super(SimpleCNN, self).__init__()
        
        # allow constructing with different channel configs to match checkpoints
        self.base_channels = base_channels
        c1 = self.base_channels
        c2 = self.base_channels * 2
        c3 = self.base_channels * 4
        
        self.conv1 = nn.Conv2d(3, c1, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        
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
    checkpoint = torch.load(model_path, map_location=device)

    # Handle state_dict wrapping
    state = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    new_state = {}
    for k, v in state.items():
        new_state[k.replace("module.", "")] = v

    # Detect base channel size from checkpoint conv1 if available
    base_channels = 32
    try:
        conv1_w = new_state.get("conv1.weight")
        if isinstance(conv1_w, torch.Tensor) and conv1_w.ndim == 4:
            # shape: [out_channels, in_channels, kH, kW]
            base_channels = int(conv1_w.shape[0])
    except Exception:
        pass

    model = SimpleCNN(num_classes=2, base_channels=base_channels)
    model.load_state_dict(new_state, strict=False)
    model.to(device).eval()
    print(f"✅ Safety filter model loaded (base_channels={base_channels})")
    return model