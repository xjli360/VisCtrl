import argparse
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDIMScheduler

from visctrl.diffuser_utils import VisCtrlPipeline
from visctrl.visctrl_utils import AttentionBase, regiter_attention_editor_diffusers

from torchvision.utils import save_image
from pytorch_lightning import seed_everything

from visctrl.visctrl import SelfAttentionControl
from torchvision.io import read_image

import numpy as np
from segment_anything import SamPredictor, sam_model_registry
import torch
import cv2
from PIL import Image

def load_image_device(image_path, device):
    image = read_image(image_path)
    image = image[:3].unsqueeze_(0).float() / 127.5 - 1.  # [-1, 1]
    image = F.interpolate(image, (512, 512))
    image = image.to(device)
    return image

def load_image(image_path, device):
    image = read_image(image_path)
    image = image[:3].unsqueeze_(0).float() / 127.5 - 1.  # [-1, 1]
    image = F.interpolate(image, (512, 512))
    image = image.to(device)
    return image

def mask_gen_img(img_path, mask_path, output_path, save_cropped_mask=False, kernel_size=2):
    # 用mask.png去截取gen.jpg对应区域的图片，并将其resize为（224，224）
    import cv2
    import numpy as np

    # 读取原始图像和mask
    gen_img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # cv2.imwrite('./tmp.jpg', image)
    # 确保mask与原始图像相同大小
    mask = cv2.resize(mask, (gen_img.shape[1], gen_img.shape[0]))
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.erode(mask, kernel, iterations=1)
    # mask = cv2.GaussianBlur(mask, (3, 3), 0)

    # 使用掩码提取对应区域
    masked_img = cv2.bitwise_and(gen_img, gen_img, mask=mask)

    # 将mask的非掩码区域变为白色
    non_masked_area = cv2.bitwise_not(mask)
    masked_img[non_masked_area != 0] = [255, 255, 255]

    # 找到对应区域的边界框
    x, y, w, h = cv2.boundingRect(mask)

    # 截取对应区域
    cropped_img = masked_img[y:y + h, x:x + w]


    # 获取原始图像的大小
    height, width = cropped_img.shape[:2]

    # 设置画布大小
    canvas_size = max(height, width)

    # 创建白色背景的画布
    canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255

    # 计算图像放置位置
    start_y = (canvas_size - height) // 2
    end_y = start_y + height
    start_x = (canvas_size - width) // 2
    end_x = start_x + width

    # 将图像放置在中心位置
    canvas[start_y:end_y, start_x:end_x] = cropped_img

    # 保存结果
    cv2.imwrite(output_path, canvas)

    if save_cropped_mask:
        canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 0
        start_y = (canvas_size - height) // 2
        end_y = start_y + height
        start_x = (canvas_size - width) // 2
        end_x = start_x + width

        # 将图像放置在中心位置
        canvas[start_y:end_y, start_x:end_x] = cropped_img

        # 保存结果
        cv2.imwrite(output_path, canvas)

def mask_ref_img(img_path, mask_path, output_path, save_cropped_mask=False, kernel_size=2):
    # 用mask.png去截取gen.jpg对应区域的图片，并将其resize为（224，224）
    import cv2
    import numpy as np

    # 读取原始图像和mask
    gen_img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # cv2.imwrite('./tmp.jpg', image)
    # 确保mask与原始图像相同大小
    mask = cv2.resize(mask, (gen_img.shape[1], gen_img.shape[0]))
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    # mask = cv2.erode(mask, kernel, iterations=1)
    # mask = cv2.GaussianBlur(mask, (3, 3), 0)

    # 使用掩码提取对应区域
    masked_img = cv2.bitwise_and(gen_img, gen_img, mask=mask)

    # 将mask的非掩码区域变为白色
    non_masked_area = cv2.bitwise_not(mask)
    masked_img[non_masked_area != 0] = [255, 255, 255]

    # 找到对应区域的边界框
    x, y, w, h = cv2.boundingRect(mask)

    # 截取对应区域
    cropped_img = masked_img[y:y + h, x:x + w]


    # 获取原始图像的大小
    height, width = cropped_img.shape[:2]

    # 设置画布大小
    canvas_size = max(height, width)

    # 创建白色背景的画布
    canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255

    # 计算图像放置位置
    start_y = (canvas_size - height) // 2
    end_y = start_y + height
    start_x = (canvas_size - width) // 2
    end_x = start_x + width

    # 将图像放置在中心位置
    canvas[start_y:end_y, start_x:end_x] = cropped_img

    # 保存结果
    cv2.imwrite(output_path, canvas)

    if save_cropped_mask:
        canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 0
        start_y = (canvas_size - height) // 2
        end_y = start_y + height
        start_x = (canvas_size - width) // 2
        end_x = start_x + width

        # 将图像放置在中心位置
        canvas[start_y:end_y, start_x:end_x] = cropped_img

        # 保存结果
        cv2.imwrite(output_path, canvas)

def gen_box(file_name):
    with open(file_name) as f:
        content = json.load(f)
    point_list = content.get("shapes")[0].get('points')
    max_x = max(point_list[0][0], point_list[1][0], point_list[2][0], point_list[3][0])
    max_y = max(point_list[0][1], point_list[1][1], point_list[2][1], point_list[3][1])
    return np.array([point_list[0][0], point_list[0][1], max_x, max_y])

def sam_img(img_path, input_box, save_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    predictor.set_image(img)

    masks, _, _ = predictor.predict(point_coords=None, point_labels=None, box=input_box[None, :],
                                    multimask_output=False)
    mask_gray = np.where(masks, 255, 0).astype(np.uint8)
    mask_image = Image.fromarray(mask_gray[0], mode='L')
    mask_image.save(save_path)


def lang_sam_img(img_path, prompt, save_path):
    import torch
    torch.cuda.set_device(1)  # set the GPU device
    from PIL import Image
    from lang_sam import LangSAM
    lang_sam = LangSAM()


    image_pil = Image.open(img_path).convert("RGB")
    masks, boxes, phrases, logits = lang_sam.predict(image_pil, prompt)
    mask_gray = np.where(masks, 255, 0).astype(np.uint8)
    mask_image = Image.fromarray(mask_gray[0], mode='L')
    mask_image.save(save_path)


seed = 42
seed_everything(seed)
sam = sam_model_registry["vit_h"](checkpoint="/workspace/hf_data/sam/sam_vit_h_4b8939.pth")
sam.to(device='cuda:0')
predictor = SamPredictor(sam)


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

first_flag = True

def vis_ctrl(tar_img_path, src_img_path, injection_count,
             src_prompt, tar_prompt,save_path=None):

    first_flag = True
    for inj_num in range(injection_count):
        if first_flag:
            attention_injection(tar_img_path=tar_img_path, src_img_path=src_img_path, g_scale=6, num_step=10, step=2, layer=1,
                                injection_count=inj_num, src_prompt=src_prompt, tar_prompt=tar_prompt,save_path=save_path)
        else:
            attention_injection(tar_img_path=tar_img_path, src_img_path=src_img_path, g_scale=4, num_step=5, step=1, layer=1,
                                injection_count=inj_num, src_prompt=src_prompt, tar_prompt=tar_prompt,save_path=save_path)
        first_flag = False

    print('vis ctrl done!')


def attention_injection(tar_img_path, src_img_path, g_scale, num_step, step, layer, injection_count,
                        src_prompt, tar_prompt, save_path=None):

    torch.cuda.set_device(3)  # set the GPU device
    model_path = "/workspace/hf_data/stable-diffusion-v1-5"
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                              set_alpha_to_one=False)
    model = VisCtrlPipeline.from_pretrained(model_path, scheduler=scheduler).to(device)

    global first_flag
    if first_flag:
        tar_img_path = tar_img_path
    else:
        tar_img_path = save_path
    # target image
    tar_image = load_image(tar_img_path, device)

    # source image
    source_image = load_image(src_img_path, device)

    source_prompt = src_prompt
    target_prompt = tar_prompt

    start_code_tar, latents_list_tar = model.invert(tar_image,
                                                    target_prompt,
                                                    guidance_scale=g_scale,
                                                    num_inference_steps=num_step,
                                                    return_intermediates=True)

    start_code_src, latents_list_src = model.invert(source_image,
                                                    source_prompt,
                                                    guidance_scale=g_scale,
                                                    num_inference_steps=num_step,
                                                    return_intermediates=True)

    editor = SelfAttentionControl(step, layer)
    regiter_attention_editor_diffusers(model, editor)

    prompts = [source_prompt, source_prompt, target_prompt, target_prompt]
    start_code = torch.cat([start_code_src, start_code_src, start_code_tar, start_code_tar])

    image_masactrl = model(prompts,
                           latents=start_code,
                           guidance_scale=g_scale,
                           tar_intermediate_latents=latents_list_tar,
                           src_intermediate_latents=latents_list_src,
                           num_inference_steps=num_step,
                           )

    # save the synthesized image
    out_image = torch.cat([image_masactrl[0:1],
                           image_masactrl[2:3],
                           # image_fixed,
                           image_masactrl[-1:]], dim=0)
    # save_image(out_image,save_path))
    save_image(image_masactrl[-1:], save_path)
    print("Syntheiszed images are saved in", save_path)

def getArg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tar_prompt', type=str, required=True)
    parser.add_argument('--ref_prompt', type=str, required=True)
    parser.add_argument('--iteration_num', type=int, required=True)

    parser.add_argument('--tar_img_path', type=str, required=True)
    parser.add_argument('--tar_img_mask_path', type=str, required=True)
    parser.add_argument('--ref_img_path', type=str, required=True)
    parser.add_argument('--ref_img_mask_path', type=str, required=True)

    return parser.parse_args()


args = getArg()
# ************************************************************
# Parameter
# ************************************************************
tar_prompt = args.tar_prompt
ref_prompt = args.ref_prompt
vis_ctrl_it_num = args.iteration_num
is_crop_img = True


# ************************************************************
# Segment objects in target images, such as glasses, cars, etc...
# ************************************************************
gen_img_path = args.tar_img_path
gen_mask_path = args.tar_img_mask_path

gen_box_path = args.tar_img_mask_path + './tar_img_box.json'
sam_img(gen_img_path, gen_box(gen_box_path), gen_mask_path)


crop_gen_path = args.tar_img_path + './tar_img_cropped.jpg'
mask_gen_img(gen_img_path, gen_mask_path, crop_gen_path)
print('crop tar img done!')


# ************************************************************
# Segment objects in reference images, such as glasses, cars, etc...
# ************************************************************
# get ref img
raw_ref_img_path = args.ref_img_path
ref_mask_path = args.ref_img_mask_path

ref_box_path = args.ref_img_mask_path + './ref_img_box.json'
sam_img(gen_img_path, gen_box(gen_box_path), gen_mask_path)

# if is_crop_img:
#     lang_sam_img(raw_ref_img_path,ref_prompt,ref_mask_path)

crop_img_path = args.ref_img_path + './ref_img_cropped.jpg'

mask_gen_img(raw_ref_img_path, ref_mask_path, crop_img_path, kernel_size=2)
print('crop ref img done!')

# ************************************************************
# VisCtrl
# ************************************************************
gen_new_img_path = "./tar_cropped_new.jpg"
vis_ctrl(crop_gen_path, crop_img_path,vis_ctrl_it_num,ref_prompt,tar_prompt,save_path=gen_new_img_path)
gen_img = cv2.imread(gen_new_img_path)

tar_src_img = cv2.imread(gen_img_path)

gen_new = cv2.resize(gen_img, tar_src_img.shape)
cv2.imwrite(gen_new_img_path, gen_new)
# inpainting back to raw img
new_gen_mask_path = "./tar_cropped_new_mask.jpg"
print('VisCtrl done!')

# ***********************************************************
# Segment the objects in the generated image and then insert them back into the original image
# ************************************************************
sam_img(gen_new_img_path, gen_box(gen_box_path), new_gen_mask_path)
# lang_sam_img(gen_new_img_path,ref_prompt,new_gen_mask_path)
print('gen new mask done!')

new_mask = cv2.imread(new_gen_mask_path, cv2.IMREAD_GRAYSCALE)
mask = cv2.imread(gen_mask_path, cv2.IMREAD_GRAYSCALE)
x1, y1, w1, h1 = cv2.boundingRect(new_mask)

new_cropped_img = gen_new[y1:y1 + h1, x1:x1 + w1]
new_cropped_mask = new_mask[y1:y1 + h1, x1:x1 + w1]


# Read the original image and original mask
raw_img = cv2.imread(gen_img_path)
gen_mask = cv2.imread(gen_mask_path, cv2.IMREAD_GRAYSCALE)
gen_mask = cv2.resize(gen_mask, (raw_img.shape[1], raw_img.shape[0]))

tmp_img = cv2.cvtColor(gen_mask, cv2.COLOR_GRAY2BGR)
tmp_mask = gen_mask.copy()

# Find the bounding box of the corresponding area
# cv2.imwrite('./tmp.jpg', new_cropped_img)
x, y, w, h = cv2.boundingRect(gen_mask)


new_cropped_mask = cv2.resize(new_cropped_mask, (w, h))
tmp_mask[y:y + h, x:x + w] = new_cropped_mask

new_cropped_img = cv2.resize(new_cropped_img, (w, h))
tmp_img[y:y + h, x:x + w] = new_cropped_img

ret, mask = cv2.threshold(tmp_mask, 100, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)
old_gen_img_bg = cv2.bitwise_and(raw_img, raw_img, mask=mask_inv)
new_gen_img_fg = cv2.bitwise_and(tmp_img, tmp_img, mask=mask)
merged_img = cv2.add(old_gen_img_bg, new_gen_img_fg)

cv2.imwrite('./res/tar_new.jpg', merged_img)
cv2.imwrite('./res/tar_new_mask.jpg', mask)
print('all done!')

