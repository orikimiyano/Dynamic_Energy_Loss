import numpy as np
import os                
import nibabel as nib    
import imageio
import cv2
import warnings
import skimage.io as io

warnings.filterwarnings("ignore")

from PIL import Image


def nii_to_image(niifile_i):
    filenames = os.listdir(filepath_i)  
    slice_trans = []
 
    for f in filenames:
        
        img_path = os.path.join(filepath_i, f)
        img = nib.load(img_path)                
        img_fdata = img.get_fdata()
        fname = f.replace('.nii.gz','')           
        img_f_path = os.path.join(imgfile_i, fname)
        
        if not os.path.exists(img_f_path):
            os.mkdir(img_f_path)                
 
        (x,y,z) = img.shape
        for i in range(z):                      
            silce = img_fdata[:, :, i]
            silce = silce.astype(np.float32)
            silce = silce / 255.0
            silce = cv2.cvtColor(silce,cv2.COLOR_GRAY2BGR)
            imageio.imwrite(os.path.join(img_f_path,'{}.png'.format(i)), silce)

def nii_to_label(niifile_l):
    filenames = os.listdir(filepath_l)  
    slice_trans = []
 
    for f in filenames:
        
        img_path = os.path.join(filepath_l, f)
        img = nib.load(img_path)                
        img_fdata = img.get_fdata()
        fname = f.replace('.nii.gz','')           
        img_f_path = os.path.join(imgfile_l, fname)
        
        if not os.path.exists(img_f_path):
            os.mkdir(img_f_path)                
 
        (x,y,z) = img.shape
        for i in range(z):                      
            silce = img_fdata[:, :, i]
            cv2.imwrite(os.path.join(img_f_path,'{}.png'.format(i)), silce)
            mask = Image.open(img_f_path+'/'+str(i)+'.png').convert('L')

            mask.putpalette([0, 0, 0,
                             128, 0, 0,
                             0, 128, 0,
                             128, 128, 0,
                             0, 0, 128,
                             128, 0, 128,
                             0, 128, 128,
                             128, 128, 128,
                             64, 0, 0,
                             192, 0, 0,
                             64, 128, 0,  ##
                             192, 128, 0,
                             64, 0, 128,
                             192, 0, 128,
                             64, 128, 128,
                             192, 128, 128,
                             0, 64, 0,
                             128, 64, 0,
                             0, 192, 0,
                             128, 192, 0])

            mask.save(img_f_path+'/'+str(i)+".png")
                                                
 
if __name__ == '__main__':
    filepath_l = './data/gt_nii'
    imgfile_l = './data/test_turth'
    #filepath_i = './data/MR'
    #imgfile_i = './data/MR2D'
    #nii_to_image(filepath_i)
    nii_to_label(filepath_l)
