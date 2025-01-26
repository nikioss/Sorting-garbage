import os
import pandas as pd
from ast import literal_eval
import torch
import cv2
from torch.utils.data import Dataset


class WasteDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

        # Фильтрация записей, если файл отсутствует
        self.annotations = self.annotations[self.annotations['file_name'].apply(
            lambda x: os.path.exists(os.path.join(root_dir, x))
        )]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        file_name = row['file_name']
        category_id = row['category_id'] - 1
        bbox = literal_eval(row['bbox'])

        # Преобразование bbox в формат [x_min, y_min, x_max, y_max]
        bbox = [
            bbox[0],  
            bbox[1],    
            bbox[0] + bbox[2],   
            bbox[1] + bbox[3]        
        ]
        
        img_path = os.path.join(self.root_dir, file_name)
        image = cv2.imread(img_path)

        if image is None:
            raise FileNotFoundError(f"Image not found at {img_path}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            transformed = self.transform(image=image, bboxes=[bbox], class_labels=[category_id])
            image = transformed["image"]
            bbox = transformed["bboxes"][0]
            category_id = transformed["class_labels"][0]

        bbox_tensor = torch.tensor(bbox, dtype=torch.float32)
        category_tensor = torch.tensor(category_id, dtype=torch.long)

        return image, bbox_tensor, category_tensor
