import argparse

def get_args():
    parser = argparse.ArgumentParser(description="SAM test")
    ########## base options ##########
    parser.add_argument('--output_dir', default='outputs')
    parser.add_argument('--device', default='cuda')
    ########## sam options ##########
    parser.add_argument('--mode', default='auto', help='[auto, prompt]')
    parser.add_argument('--model_type', default='vit_h')
    parser.add_argument('--checkpoint', default='download_model/sam_vit_h_4b8939.pth')
    ########## prompt options ##########
    parser.add_argument('--point', default=[[75,75]], type=list)
    parser.add_argument('--point_label', default=[1], type=list)
    parser.add_argument('--box', default=[0,0,150,150], type=list)
    ########## auto options ##########
    parser.add_argument('--points_per_side', default=32, type=int)
    parser.add_argument('--points_per_batch', default=64, type=int)
    parser.add_argument('--pred_iou_thresh', default=0.88, type=float)
    parser.add_argument('--stability_score_thresh', default=0.95, type=float)
    parser.add_argument('--stability_score_offset', default=1.0, type=float)
    parser.add_argument('--box_nms_thresh', default=0.7, type=float)
    parser.add_argument('--crop_n_layers', default=0, type=int)
    parser.add_argument('--crop_nms_thresh', default=0.7, type=float)
    parser.add_argument('--crop_overlap_ratio', default=512/1500, type=float)
    parser.add_argument('--crop_n_points_downscale_factor', default=1, type=int)
    parser.add_argument('--min_mask_region_area', default=0, type=int)
    parser.add_argument('--output_mode', default='binary_mask', type=str)
    ########## dataset options ##########
    parser.add_argument('--data_type', default='sperm', help='[test_image, sperm]')
    parser.add_argument('--image_path', default='datasets/groceries.jpg')
    parser.add_argument('--sperm_path', default='datasets/771_1stframe_img.pkl')
    parser.add_argument('--sperm_id', default='027')

    args = parser.parse_args()
    return args