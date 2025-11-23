# Copyright 2025 antillia.com Toshiyuki Arai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ImageMaskDatasetGenerator.py
# 2025/11/23 Modified to generate PNG files.

import os
import shutil
import glob
import nibabel as nib
import numpy as np
import traceback
import cv2

class ImageMaskDatasetGenerator:

  def __init__(self, 
               images_dir  = "./", 
               masks_dir   = "./",
               output_dir = "./master", 
               resize     = 512):
    
    self.images_dir = images_dir 
    self.masks_dir  = masks_dir
    self.W          = resize
    self.H          = resize

    if not os.path.exists(self.images_dir):
      raise Exception("Not found " + images_dir)   

    if os.path.exists(output_dir):
      shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    self.output_images_dir = os.path.join(output_dir, "images")
    self.output_masks_dir  = os.path.join(output_dir, "masks")

    os.makedirs(self.output_images_dir)
    os.makedirs(self.output_masks_dir)

    self.seed    = 137
    self.angle   = 0  #cv2.ROTATE_90_COUNTERCLOCKWISE

    self.RESIZE    = (resize, resize)
    self.file_format= ".png"


  def generate(self):
    index = 10000
   
    mask_files = glob.glob(self.masks_dir + "/*_seg.nii.gz")
    mask_files = sorted(mask_files)   
    
    l = len(mask_files)
    for i in range(l):
      #Take the last label_nii.gz file.
      mask_file = mask_files[i]
      
      self.generate_mask_files(mask_file,    index +i)

      image_file = mask_file.replace("_seg", "_norm")
      #end_points  = mask_file.replace("_seg", "_seg_endpoints")
      self.generate_image_files(image_file,   index+i) 


  def generate_image_files(self, niigz_file, index):
    nii = nib.load(niigz_file)
    fdata  = nii.get_fdata()
   
    w, h, d = fdata.shape
    print("=== image shape {}".format(fdata.shape))
    for i in range(d):
      img = fdata[:,:, i]
      filename  = str(index) +"_" +str(i) + self.file_format
      filepath  = os.path.join(self.output_images_dir, filename)
      corresponding_mask_file = os.path.join(self.output_masks_dir, filename)
      # Does the corresponding non empty mask file exist?
      if os.path.exists(corresponding_mask_file):
        img = cv2.resize(img, self.RESIZE)
        img = cv2.rotate(img, self.angle)
        img = img*255
        cv2.imwrite(filepath, img)
        print("=== Saved {}".format(filepath))
      
  def generate_mask_files(self, niigz_file, index ):
    nii = nib.load(niigz_file)
    fdata  = nii.get_fdata()
   
    w, h, d = fdata.shape
    print("=== mask shape {}".format(fdata.shape))
    #input("HIT any key")
    for i in range(d):
      img = fdata[:,:, i]
      filename  = str(index) + "_" + str(i) + self.file_format

      filepath  = os.path.join(self.output_masks_dir, filename)
      
      #Skip empty all black mask
      if img.any() >0:
        img = cv2.resize(img, self.RESIZE)
        img = cv2.rotate(img, self.angle)
        img = self.colorize_mask(img) 
        cv2.imwrite(filepath, img)
        print("--- Saved {}".format(filepath))
 
  
  def colorize_mask(self, mask):
    h, w = mask.shape[:2]
    colorized = np.zeros((h, w, 3), dtype=np.float32)

    #      BGR color
    # Please see https://www.nature.com/articles/s41597-024-03469-9
    LV = (  0,   0,  255)    #1: red
    RV = (255,   0,   0)     #2: blue
    LA = (  0, 255,  255)    #3: yellow
    RA = (255, 255,   0)     #4: cyan
    AD = (  0, 255,   0)     #5: green    
    PA = ( 255, 255, 255)    #6: white
    SVC = (255,  0, 255)     #7: mazenda    
    IVC = (128, 128, 128)    #8: gray

    colorized[np.equal(mask, 1)] = LV
    colorized[np.equal(mask, 2)] = RV
    colorized[np.equal(mask, 3)] = LA
    colorized[np.equal(mask, 4)] = RA
    colorized[np.equal(mask, 5)] = AD
    colorized[np.equal(mask, 6)] = PA
    colorized[np.equal(mask, 7)] = SVC
    colorized[np.equal(mask, 8)] = IVC

    return colorized

  
if __name__ == "__main__":
  try:
    images_dir  = "./cropped_norm/"
    masks_dir   = "./cropped_norm/"
    resize      = 512
    augmentation = False
    output_dir  = "./Whole-Heart-master/"
    input("Hit any key to start!")
    generator = ImageMaskDatasetGenerator(resize=resize,
                                             images_dir  = images_dir, 
                                          masks_dir   = masks_dir,
                                          output_dir = output_dir)
                                      
    generator.generate()
  except:
    traceback.print_exc()

 
