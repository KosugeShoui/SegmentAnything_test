import numpy as np
import matplotlib.pyplot as plt

def show_img(img, save_dir):
    plt.figure(figsize=(10,10))
    plt.imshow(img)
    plt.axis('off')
    plt.savefig(f'{save_dir}/img.png')
    
def show_img_with_prompt(img, save_dir, point=None, point_label=None, box=None, mask=None):
    plt.figure(figsize=(10,10))
    plt.imshow(img)
    if type(point) == np.ndarray: show_points(point, point_label, plt.gca())
    if type(box)   == np.ndarray: show_box(box, plt.gca())
    plt.axis('off')
    plt.savefig(f'{save_dir}/img_with_prompt.png')

def show_img_with_prompt_and_mask(img, masks, save_dir, point=None, point_label=None, box=None):
    for i, mask in enumerate(masks):
        plt.figure(figsize=(10,10))
        plt.imshow(img)
        show_mask(mask, plt.gca())
        if type(point) == np.ndarray: show_points(point, point_label, plt.gca())
        if type(box)   == np.ndarray: show_box(box, plt.gca())
        plt.title(f"Mask {i+1}", fontsize=18)
        plt.axis('off')
        plt.savefig(f'{save_dir}/img_with_prompt_and_mask_{i}.png')

def show_img_with_mask(img, mask, save_dir):
    plt.figure(figsize=(10,10))
    plt.imshow(img)
    show_anns(mask)
    plt.axis('off')
    plt.savefig(f'{save_dir}/img_with_mask.png')

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    
    
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)