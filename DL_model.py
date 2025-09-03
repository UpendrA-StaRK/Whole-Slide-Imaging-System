import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import cv2
import time
class ResNetFocusRegressor(nn.Module):
    def _init_(self):
        super()._init_()
        self.backbone = models.resnet34(weights=None)
        num_feats = self.backbone.fc.in_features  
        self.backbone.fc = nn.Identity()          

        self.reg_head = nn.Sequential(
            nn.Linear(num_feats, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        feats = self.backbone(x)      
        out = self.reg_head(feats)     
        return out

class InlineFocusDataset(Dataset):
    """Dataset for inline focus prediction with known focus distances"""
    def _init_(self, image_dir,transform=None):
        self.image_dir = image_dir
        self.transform = transform
        # Load focus distances from file
        self.image_files = []
        self.distances = []
        for file_name in os.listdir(image_dir):
            path = os.path.join(image_dir, file_name)
            if os.path.isfile(path):
                name = os.path.splitext(file_name)[0] 
                parts = name.split('_')
                if len(parts) == 4:
                    try:
                        z = float(parts[2])
                        self.image_files.append(file_name)
                        self.distances.append(z*1000)
                    except ValueError:
                        print(f"Skipping {file_name}: invalid z value")
    
    def _len_(self):
        return len(self.image_files)
    
    def _getitem_(self, idx):
        # Load image
        img_path = os.path.join(self.image_dir,self.image_files[idx])
        image = Image.open(img_path)
        if image.mode == 'L':
            image = image.convert('RGB')
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            # Default transformation if none provided
            image = transforms.ToTensor()(image)
        
        # Get focus distance
        distance = torch.tensor([self.distances[idx]], dtype=torch.float32)
        
        return image, distance

def train_inline_focus_model(model, train_loader, val_loader, num_epochs=10, lr=0.0001):
    """Train the inline focus prediction model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()  
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    best_val_loss = float('inf')
    train_arr,val_arr = [],[]
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        start_time = time.time()
        
        for images, distances in train_loader:
            images = images.to(device)
            distances = distances.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Calculate loss
            loss = torch.sqrt(criterion(outputs, distances))
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_arr.append(train_loss)
        epoch_time = time.time() - start_time
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, distances in val_loader:
                images = images.to(device)
                distances = distances.to(device)
                
                outputs = model(images)
                loss = torch.sqrt(criterion(outputs, distances))
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        val_arr.append(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, ' 
              f'Val Loss: {val_loss:.6f}, Time: {epoch_time:.2f}s')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'resnet_best_focus_model.pth')
    plt.figure()
    epochs = range(1, num_epochs+1)
    plt.plot(epochs, train_arr, label='Train Loss')
    plt.plot(epochs, val_arr,   label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.title('Loss Curves');
    plt.savefig('resnet_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
            
    return model

def predict_focus_distance(model, image_path):
    """Predict focus distance for a single image"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Load image
    image = Image.open(image_path)
    if image.mode == 'L':
        image = image.convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict focus distance
    with torch.no_grad():
        start_time = time.time()
        distance = model(tensor).item()
        inference_time = time.time() - start_time
    
    return distance, inference_time
    
def main():
    torch.backends.cudnn.benchmark = True
    
    # Create model
    model = ResNetFocusRegressor()

    train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
        
    val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Example data paths (replace with actual paths)
    train_image_dir = "Train_New"
    val_image_dir = 'Data/Val'
    
    # Create datasets and dataloaders
    train_dataset = InlineFocusDataset(
        train_image_dir,
        transform=train_transform
    )
    
    val_dataset = InlineFocusDataset(
        val_image_dir,
        transform=val_transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=12,pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=12,pin_memory=True)
    trained_model = train_inline_focus_model(model, train_loader, val_loader, num_epochs=40)
    
if _name_ == "_main_":
    main()
