from jittor.dataset.dataset import Dataset
import jittor.transform as transform
import cv2
import os

class SymbolDataset(Dataset):
    def __init__(self, data_path,pic_size):
        super().__init__()
        
        pic_path = os.path.join(data_path, str(pic_size))
        self.images = []
        
        for img_path in os.listdir(pic_path):
            image_path = os.path.join(pic_path, img_path)
        
            image = cv2.imread(image_path).astype('uint8')
            self.images.append(image)  
        self.flip_trans=transform.RandomHorizontalFlip(0.5)
        self.img_normal=transform.ImageNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img = self.images[index]
        img=transform.to_pil_image(img)
        img=self.flip_trans(img)
        img=transform.to_tensor(img)
        img=self.img_normal(img)
        return img