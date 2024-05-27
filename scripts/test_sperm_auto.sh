python test_sam.py \
    --device cuda \
    --mode auto \
    --model_type vit_h \
    --checkpoint download_model/sam_vit_h_4b8939.pth \
    --data_type sperm \
    --sperm_path datasets/771_1stframe_img.pkl \
    --sperm_id 026