from PIL import Image
import sys
import os
        
if __name__ == '__main__':
    path= "/home/user/Desktop/stylegan/color_symbol_7k"
    path_save="/home/user/Desktop/stylegan/data/symbol"
    for pic_size in [8,16,32,64,128]:
        if not os.path.exists(os.path.join(path_save,str(pic_size))):
            os.mkdir(os.path.join(path_save,str(pic_size)))
    for pic in os.listdir(path):
        img = Image.open(os.path.join(path,pic))
        for pic_size in [8,16,32,64]:    
            img_resize=img.resize((pic_size, pic_size), Image.LANCZOS)
            img_resize.save(os.path.join(path_save,str(pic_size),pic))
        img.save(os.path.join(path_save,str(128),pic))
            