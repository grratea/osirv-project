# import os
# from pathlib import Path

# base     = Path(r"C:\Users\Tea\Desktop\EG1800")
# img_dir  = base / "images_data_crop"
# mask_dir = base / "GT_png"

# imgs  = sorted(img_dir.glob("*.*"))[:5]
# masks = sorted(mask_dir.glob("*.*"))[:5]

# print("SLIKE:")
# for f in imgs:  print(" ", f.name)

# print("\nMASKE:")
# for f in masks: print(" ", f.name)

import torch; 
print('CUDA:', torch.cuda.is_available()); 
print('GPU:', torch.cuda.get_device_name(0))