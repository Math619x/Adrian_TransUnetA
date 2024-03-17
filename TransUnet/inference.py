import os
import cv2
import torch
import numpy as np
import datetime

# Additional Scripts
from train_transunet import TransUNetSeg

from utils.utils import thresh_func
from config import cfg


class SegInference:
    def __init__(self, model_path, device):
        self.device = device
        self.transunet = TransUNetSeg(device)
        self.transunet.load_model(model_path)

        if not os.path.exists('./results'):
            os.mkdir('./results')

    def read_and_preprocess(self, p):
        
        img = np.load(p)
        img_torch = img.transpose((2, 0, 1))
        img_torch = np.expand_dims(img_torch, axis=0)
        img_torch = torch.from_numpy(img_torch.astype('float32')).to(self.device)

        return img, img_torch

    def save_preds(self, preds):
        folder_path = './results/' + str(datetime.datetime.utcnow()).replace(' ', '_')

        os.mkdir(folder_path)
        for name, pred_mask in preds.items():
            cv2.imwrite(f'{folder_path}/{name.replace(".npy",".png")}', pred_mask)
            print(f'Image saved in: {folder_path}/{name.replace(".npy",".png")}')

    def infer(self, path, merged=False, save=True):
        path = [path] if isinstance(path, str) else path
        
        path = path if path[0].endswith('.npy') else [os.path.join(path[0],filename) for filename in os.listdir(path[0])]

        preds = {}
        for p in path:
            file_name = p.split('/')[-1]
            img, img_torch = self.read_and_preprocess(p)
            with torch.no_grad():
                pred_mask = self.transunet.model(img_torch)
                pred_mask = torch.sigmoid(pred_mask)
                pred_mask = pred_mask.detach().cpu().numpy()
    
            color_map = [
            (0, 0, 0),       # Clase 0: negro
            (0, 0, 255),     # Clase 1: azul
            (0, 255, 0),     # Clase 2: verdes
            (255, 0, 0),     # Clase 3: rojo
            (0, 255, 255),   # Clase 4: amarillo
            # Agrega más colores si tienes más clases
            ]
            
            pred_mask = np.argmax(pred_mask, axis=1)
            pred_mask = pred_mask[0]
            
            # Crea una imagen en color vacía con las mismas dimensiones que la máscara
            color_mask = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)

            # Asigna el color correspondiente a cada clase en la imagen en color
            for label, color in enumerate(color_map):
                color_mask[pred_mask == label] = color

            if merged:
                pred_mask = cv2.bitwise_and(img, img, mask=pred_mask.astype('uint8'))

            preds[file_name] = color_mask

        if save:
            self.save_preds(preds)

        return preds
