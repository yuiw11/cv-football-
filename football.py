import os
import json
import cv2
import pickle
import numpy as np
from IPython.display import Video
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import pandas as pd


class FootballDataset(Dataset):
    def __init__(self, root):
        self.frames = pd.DataFrame()
        self.image_paths = []
        folders = ['Match_1824_1_0_subclip_3', 'Match_1951_1_0_subclip', 'Match_2022_3_0_subclip']
        frame_counter = 0  
        for folder in folders:
            folder_path = os.path.join(root, folder)
            for file in os.listdir(folder_path):
                if file.endswith('.json'):
                    for json_path in Path(folder_path).rglob('*.json'):
                        with open(json_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            frame_labels = [annotation for annotation in data['annotations'] if annotation['category_id'] == 4]
                            data = []
                            for label in frame_labels:
                                image_id = label['image_id'] + frame_counter 
                                bbox = label['bbox']
                                jersey_number = label.get('attributes', {}).get('jersey_number', '-1')
                                data.append({'image_id': image_id, 'bbox': bbox, 'jersey_number': jersey_number})
                            frame_counter += max([label['image_id'] for label in frame_labels], default=0) 
                            df = pd.DataFrame(data).set_index('image_id')  
                            self.frames = pd.concat([self.frames, df])        

                
                elif file.endswith('.mp4'):
                    video_path = os.path.join(folder_path, file)
                    cap = cv2.VideoCapture(video_path)
                    output_folder = os.path.join(folder_path, "frames")
                    os.makedirs(output_folder, exist_ok=True)
                    frame_idx = 0
                    while cap.isOpened():
                        ret, image = cap.read()
                        if not ret:
                            break
                        frame_path = os.path.join(output_folder, f"frame_{frame_idx}.jpg")
                        cv2.imwrite(frame_path, image)
                        self.image_paths.append(frame_path)
                        frame_idx += 1
                    cap.release()


    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        jersey_numbers = []
        bboxes = []
        if idx in self.frames.index:
            rows = self.frames.loc[[idx]] if isinstance(self.frames.loc[idx], pd.Series) else self.frames.loc[idx]
            for _, row in rows.iterrows():
                jersey_numbers.append(row['jersey_number'])
                bboxes.append(row['bbox'])
        return img, jersey_numbers, bboxes


if __name__ == '__main__':
    root = 'football_train'
    dataset = FootballDataset(root=root)

    img, jersey_numbers, bboxes = dataset[300]
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Resize the image to a smaller size
    scale_factor = 0.5  # Adjust the scale factor as needed
    img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

    for bbox, jersey_number in zip(bboxes, jersey_numbers):
        img_height, img_width, _ = img.shape
        # Scale the bounding box coordinates to match the resized image
        scaled_bbox = [coord * scale_factor for coord in bbox]
        pt1 = (max(0, min(int(scaled_bbox[0]), img_width - 1)), max(0, min(int(scaled_bbox[1]), img_height - 1)))
        pt2 = (max(0, min(int(scaled_bbox[0] + scaled_bbox[2]), img_width - 1)), max(0, min(int(scaled_bbox[1] + scaled_bbox[3]), img_height - 1)))
        cv2.rectangle(img, pt1, pt2, (0, 0, 255), 3)  # Red rectangle
        text_position = (pt1[0], max(pt1[1] - 10, 0))
        cv2.putText(img, str(jersey_number), text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)  # Green text

    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
