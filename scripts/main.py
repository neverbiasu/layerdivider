import os
import io
import json
import numpy as np
import cv2

import gradio as gr

import modules.scripts as scripts
from modules import script_callbacks

from ldivider.ld_convertor import pil2cv, cv2pil, df2bgra
from ldivider.ld_processor import get_base, get_normal_layer, get_composite_layer, get_seg_base
from ldivider.ld_utils import save_psd, load_masks, divide_folder, load_seg_model
from ldivider.ld_segment import get_mask_generator, get_masks, show_anns

from modules.paths_internal import extensions_dir
from collections import OrderedDict

from pytoshop.enums import BlendMode




model_cache = OrderedDict()
base_dir = "layerdivider"
output_dir = os.path.join(extensions_dir, base_dir, "output")
input_dir = os.path.join(extensions_dir, base_dir, "input")
model_dir = os.path.join(extensions_dir, base_dir, "segment_model")

class Script(scripts.Script):
  def __init__(self) -> None:
    super().__init__()

  def title(self):
    return "LayerDivider"

  def show(self, is_img2img):
    return scripts.AlwaysVisible

  def ui(self, is_img2img):
    return ()

def segment_image(input_image, pred_iou_thresh, stability_score_thresh, min_mask_region_area):
    load_seg_model(model_dir)
    mask_generator = get_mask_generator(pred_iou_thresh, stability_score_thresh,min_mask_region_area, model_dir, "extension")
    masks = get_masks(pil2cv(input_image), mask_generator)
    input_image.putalpha(255)
    masked_image = show_anns(input_image, masks, output_dir)
    return masked_image

def divide_layer(divide_mode, input_image, loops, init_cluster, ciede_threshold, blur_size, layer_mode, h_split, v_split, n_cluster, alpha, th_rate, split_bg, area_th):
    if divide_mode == "segment_mode":
        return segment_divide(input_image, loops, init_cluster, ciede_threshold, blur_size, layer_mode, h_split, v_split, n_cluster, alpha, th_rate, split_bg, area_th)
    elif divide_mode == "color_base_mode":
        return color_base_divide(input_image, loops, init_cluster, ciede_threshold, blur_size, layer_mode, h_split, v_split, n_cluster, alpha, th_rate, split_bg)


def segment_divide(input_image, loops, init_cluster, ciede_threshold, blur_size, layer_mode, h_split, v_split, n_cluster, alpha, th_rate, split_bg, area_th):
    image = pil2cv(input_image)
    input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
    masks = load_masks(output_dir)

    df = get_seg_base(input_image, masks, area_th)


    base_image = cv2pil(df2bgra(df))
    image = cv2pil(image)
    if layer_mode == "composite":
        base_layer_list, shadow_layer_list, bright_layer_list, addition_layer_list, subtract_layer_list = get_composite_layer(input_image, df)
        filename = save_psd(
            input_image,
            [base_layer_list, bright_layer_list, shadow_layer_list, subtract_layer_list, addition_layer_list],
            ["base", "screen", "multiply", "subtract", "addition"],
            [BlendMode.normal, BlendMode.screen, BlendMode.multiply, BlendMode.subtract, BlendMode.linear_dodge],
            output_dir,
            layer_mode

        )
        base_layer_list = [cv2pil(layer) for layer in base_layer_list]
        divide_folder(filename, input_dir)
        return [image, base_image], base_layer_list, bright_layer_list, shadow_layer_list, filename
    elif layer_mode == "normal":
        base_layer_list, bright_layer_list, shadow_layer_list = get_normal_layer(input_image, df)
        filename = save_psd(
            input_image,
            [base_layer_list, bright_layer_list, shadow_layer_list],
            ["base", "bright", "shadow"],
            [BlendMode.normal, BlendMode.normal, BlendMode.normal],
            output_dir,
            layer_mode
        )

        divide_folder(filename, input_dir)
        return [image, base_image], base_layer_list, bright_layer_list, shadow_layer_list, filename
    else:
        return None



def color_base_divide(input_image, loops, init_cluster, ciede_threshold, blur_size, layer_mode, h_split, v_split, n_cluster, alpha, th_rate, split_bg):
    image = pil2cv(input_image)
    input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)

    df = get_base(input_image, loops, init_cluster, ciede_threshold, blur_size, h_split, v_split, n_cluster, alpha, th_rate, split_bg, False)

    base_image = cv2pil(df2bgra(df))
    image = cv2pil(image)
    if layer_mode == "composite":
        base_layer_list, shadow_layer_list, bright_layer_list, addition_layer_list, subtract_layer_list = get_composite_layer(input_image, df)
        filename = save_psd(
            input_image,
            [base_layer_list, bright_layer_list, shadow_layer_list, subtract_layer_list, addition_layer_list],
            ["base", "screen", "multiply", "subtract", "addition"],
            [BlendMode.normal, BlendMode.screen, BlendMode.multiply, BlendMode.subtract, BlendMode.linear_dodge],
            output_dir,
            layer_mode,
        )
        base_layer_list = [cv2pil(layer) for layer in base_layer_list]
        return [image, base_image], base_layer_list, bright_layer_list, shadow_layer_list, filename
    elif layer_mode == "normal":
        base_layer_list, bright_layer_list, shadow_layer_list = get_normal_layer(input_image, df)
        filename = save_psd(
            input_image,
            [base_layer_list, bright_layer_list, shadow_layer_list],
            ["base", "bright", "shadow"],
            [BlendMode.normal, BlendMode.normal, BlendMode.normal],
            output_dir,
            layer_mode,
        )
        return [image, base_image], base_layer_list, bright_layer_list, shadow_layer_list, filename
    else:
        return None

def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as LayerDivider:
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="pil")
                divide_mode = gr.Dropdown(["segment_mode", "color_base_mode"], value = "segment_mode", label="divide_mode", show_label=True)

                with gr.Accordion("Segment Settings", open=True):
                    area_th = gr.Slider(1, 100000, value=20000, step=100, label="area_threshold", show_label=True)

                with gr.Accordion("ColorBase Settings", open=True):
                    loops = gr.Slider(1, 20, value=1, step=1, label="loops", show_label=True)
                    init_cluster = gr.Slider(1, 50, value=10, step=1, label="init_cluster", show_label=True)
                    ciede_threshold = gr.Slider(1, 50, value=5, step=1, label="ciede_threshold", show_label=True)
                    blur_size = gr.Slider(1, 20, value=5, label="blur_size", show_label=True)
                    layer_mode = gr.Dropdown(["normal", "composite"], value = "normal", label="output_layer_mode", show_label=True)

                with gr.Accordion("BG Settings", open=True):
                    split_bg = gr.Checkbox(label="split bg", show_label=True)
                    h_split = gr.Slider(1, 2048, value=256, step=4, label="horizontal split num", show_label=True)
                    v_split = gr.Slider(1, 2048, value=256, step=4, label="vertical split num", show_label=True)

                    n_cluster = gr.Slider(1, 1000, value=500, step=10, label="cluster num", show_label=True)
                    alpha = gr.Slider(1, 255, value=100, step=1, label="alpha threshold", show_label=True)
                    th_rate = gr.Slider(0, 1, value=0.1, step=0.01, label="mask content ratio", show_label=True)

                submit = gr.Button(value="Create PSD")
            with gr.Row():
                with gr.Column():
                    SAM_output = gr.Image(type="pil")
                    pred_iou_thresh = gr.Slider(0, 1, value=0.8, step=0.01, label="pred_iou_thresh", show_label=True)
                    stability_score_thresh = gr.Slider(0, 1, value=0.8, step=0.01, label="stability_score_thresh", show_label=True)
                    #crop_n_layers = gr.Slider(0, 10, value=0, step=1, label="crop_n_layers", show_label=True)
                    #crop_n_points_downscale_factor = gr.Slider(1, 10, value=1, step=1, label="crop_n_points_downscale_factor", show_label=True)
                    min_mask_region_area = gr.Slider(1, 1000, value=100, step=1, label="min_mask_region_area", show_label=True)
                    segment = gr.Button(value="Segment")
                    with gr.Tab("output"):
                        output_0 = gr.Gallery()
                    with gr.Tab("base"):
                        output_1 = gr.Gallery()
                    with gr.Tab("bright"):
                        output_2 = gr.Gallery()
                    with gr.Tab("shadow"):
                        output_3 = gr.Gallery()

                    output_file = gr.File()

        submit.click(
            divide_layer,
            inputs=[divide_mode, input_image, loops, init_cluster, ciede_threshold, blur_size, layer_mode, h_split, v_split, n_cluster, alpha, th_rate, split_bg, area_th],
            outputs=[output_0, output_1, output_2, output_3, output_file]
        )
        segment.click(
            segment_image,
            inputs=[input_image, pred_iou_thresh, stability_score_thresh, min_mask_region_area],
            outputs=[SAM_output]
        )

    return [(LayerDivider, "LayerDivider", "layerdivider")]

script_callbacks.on_ui_tabs(on_ui_tabs)
