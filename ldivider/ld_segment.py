import os
import copy
import torch
import pickle
import numpy as np

from PIL import Image
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


def get_mask_generator(
    pred_iou_thresh, stability_score_thresh, min_mask_region_area, model_path, exe_mode
):
    model_cfg = "sam2_hiera_b+.yaml"
    sam2_checkpoint = os.path.join(model_path, "sam2_hiera_base_plus.pt")
    device = "cuda"
    model_type = "default"

    if exe_mode == "extension":
        from modules.safe import unsafe_torch_load, load

        torch.load = unsafe_torch_load
        sam2 = build_sam2(
            model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False
        )
        sam2.to(device=device)
        torch.load = load
    else:
        sam2 = build_sam2(
            model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False
        )
        sam2.to(device=device)

    mask_generator_2 = SAM2AutomaticMaskGenerator(
        model=sam2,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        min_mask_region_area=min_mask_region_area,
    )

    return mask_generator_2


def get_masks(image, mask_generator_2):
    if isinstance(image, np.ndarray):
        image = image.copy()
    masks = mask_generator_2.generate(image)
    return masks


def show_anns(image, masks, output_dir):
    if len(masks) == 0:
        return
    sorted_masks = sorted(masks, key=(lambda x: x["area"]), reverse=True)
    # save masks
    with open(f"{output_dir}/tmp/seg_layer/sorted_masks.pkl", "wb") as f:
        pickle.dump(sorted_masks, f)
    polygons = []
    color = []
    mask_list = []
    for mask in sorted_masks:
        m = mask["segmentation"]
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:, :, i] = color_mask[i]
        img = np.dstack((img * 255, m * 255 * 0.35))
        img = img.astype(np.uint8)

        mask_list.append(img)

    base_mask = image
    for mask in mask_list:
        base_mask = Image.alpha_composite(base_mask, Image.fromarray(mask))

    return base_mask


def show_masks(image_np, masks: np.ndarray, alpha=0.5):
    image = copy.deepcopy(image_np)
    np.random.seed(0)
    for mask in masks:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        image[mask] = image[mask] * (1 - alpha) + 255 * color.reshape(1, 1, -1) * alpha
    return image.astype(np.uint8)
