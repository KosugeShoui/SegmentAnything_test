import os
import cv2
import pickle
import numpy as np
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

from utils.options import get_args
from runner import SAMRunner
from utils.show import show_img, show_img_with_prompt, show_img_with_prompt_and_mask, show_img_with_mask

def main(args, img):
    
    sam_runner = SAMRunner(args)
    if args.mode == 'auto':
        show_img(img, args.output_dir)
        masks = sam_runner.predict_auto(img)
        show_img_with_mask(img, masks, args.output_dir)
    else:
        show_img(img, args.output_dir)
        show_img_with_prompt(img, args.output_dir, args.point, args.point_label, args.box)
        masks = sam_runner.predict_with_prompt(img)
        show_img_with_prompt_and_mask(img, masks, args.output_dir, args.point, args.point_label, args.box)

if __name__ == "__main__":

    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # set data
    if args.data_type == 'sperm':
        with open(args.sperm_path, 'rb') as f:
            img = pickle.load(f)[args.sperm_id]
    elif args.data_type == 'test_image':
        img = cv2.imread(args.image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    args.point = None if args.point==None else np.array(args.point)
    args.point_label = None if args.point_label==None else np.array(args.point_label)
    args.box = None if args.box==None else np.array(args.box)
    
    main(args, img)