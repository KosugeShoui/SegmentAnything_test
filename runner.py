import numpy as np
from typing import List
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

class SAMRunner():
    def __init__(self, args):
        super(SAMRunner, self).__init__()
        self.args = args
        
        # model
        self.sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
        self.sam.to(args.device)
        
    def predict_with_prompt(self, img) -> np.ndarray:
        self.predictor = SamPredictor(self.sam)
        self.predictor.set_image(img)
        masks, scores, logits = self.predictor.predict(
            point_coords=self.args.point,
            point_labels=self.args.point_label,
            box=self.args.box,
            multimask_output=True,
            return_logits=False,
        )
        return masks
    
    def predict_auto(self, img) -> List:
        self.predictor = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=self.args.points_per_side,
            points_per_batch=self.args.points_per_batch,
            pred_iou_thresh=self.args.pred_iou_thresh,
            stability_score_thresh=self.args.stability_score_thresh,
            stability_score_offset=self.args.stability_score_offset,
            box_nms_thresh=self.args.box_nms_thresh,
            crop_n_layers=self.args.crop_n_layers,
            crop_nms_thresh=self.args.crop_nms_thresh,
            crop_overlap_ratio=self.args.crop_overlap_ratio,
            crop_n_points_downscale_factor=self.args.crop_n_points_downscale_factor,
            min_mask_region_area=self.args.min_mask_region_area,
            output_mode=self.args.output_mode,
        )
        masks = self.predictor.generate(img)
        return masks
        