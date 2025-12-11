import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# MODEL_NAME = "meta-llama/Llama-2-7b-hf"  # Base model for embeddings
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"  # Base model for embeddings
TOPIC_LABELS = ['World', 'Sports', 'Business', 'Sci/Tech']

class TopicClassifier(nn.Module):
    def __init__(self, input_dim=4096, num_classes=4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.3)
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn3(self.fc3(x)))
        return self.fc4(x)

def load_topic_classifier(model_path="best_topic_classifier.pt"):
    """Load classifier with correct dtype + eval mode"""
    classifier = TopicClassifier().float().to(DEVICE)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    classifier.load_state_dict(checkpoint)
    classifier.eval()  # CRITICAL: BatchNorm needs eval for batch_size=1
    return classifier