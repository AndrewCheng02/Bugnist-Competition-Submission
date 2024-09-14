class BugClassifier(nn.Module):
    def __init__(self, num_classes=12):
        super(BugClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 16 * 16)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def predict_bug_type(bug_array, model_path=''):
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    if isinstance(bug_array, np.ndarray):
        bug_image = Image.fromarray(bug_array)
        bug_image = preprocess(bug_image)
        bug_image = bug_image.unsqueeze(0)
    else:
        raise TypeError("Input should be a numpy array")

    model = BugClassifier(num_classes=12)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        outputs = model(bug_image)
        _, predicted = torch.max(outputs, 1)
        label = predicted.item()

    return label