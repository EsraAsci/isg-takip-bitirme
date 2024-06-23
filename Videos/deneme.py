from ultralytics import YOLO
import cv2
import torch


device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('cihaz = ' , device)