# -*- coding: utf-8 -*-
import cv2
from PIL import Image
import numpy as np
import importlib
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import animation
import torch
from core.utils import to_tensors
 
class Inpainter(object):
    def __init__(
        self,
        device="cuda:0",
    ):
        self.device = device
        self.size = (432, 240)
        net = importlib.import_module('model.' + "e2fgvi")
        model = net.InpaintGenerator().to(device)
        data = torch.load(f"./checkpoints/E2FGVI/E2FGVI-CVPR22.pth", map_location=device)
        model.load_state_dict(data)
        model.eval()
        self.model = model
        self.ref_length = 10 
        self.num_ref = -1
        self.neighbor_stride = 5
        self.default_fps = 10
        
    def read_mask(self, mpath, size):
        masks = []
        mnames = os.listdir(mpath)
        mnames.sort()
        for mp in mnames:
            m = Image.open(os.path.join(mpath, mp))
            m = m.resize(size, Image.NEAREST)
            m = np.array(m.convert('L'))
            m = np.array(m > 0).astype(np.uint8)
            m = cv2.dilate(m,
                        cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)),
                        iterations=4)
            masks.append(Image.fromarray(m * 255))
        return masks
 
    def read_frame_from_videos(self,video_path):
        vname = video_path
        frames = []
        lst = os.listdir(vname)
        lst.sort()
        fr_lst = [vname + '/' + name for name in lst]
        for fr in fr_lst:
            image = cv2.imread(fr)
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            frames.append(image)
        return frames

    def resize_frames(self, frames, size=None):
        if size is not None:
            frames = [f.resize(size) for f in frames]
        else:
            size = frames[0].size
        return frames, size

    def get_ref_index(self, f, neighbor_ids, length):
        ref_index = []
        if self.num_ref == -1:
            for i in range(0, length, self.ref_length):
                if i not in neighbor_ids:
                    ref_index.append(i)
        else:
            start_idx = max(0, f - self.ref_length * (self.num_ref // 2))
            end_idx = min(length, f + self.ref_length * (self.num_ref // 2))
            for i in range(start_idx, end_idx + 1, self.ref_length):
                if i not in neighbor_ids:
                    if len(ref_index) > self.num_ref:
                        break
                    ref_index.append(i)
        return ref_index

    def main_worker(self, video_path, mask_path):
        # prepare datset
        frames = self.read_frame_from_videos(video_path)
        frames, size = self.resize_frames(frames, self.size)
        h, w = size[1], size[0]
        video_length = len(frames)
        imgs = to_tensors()(frames).unsqueeze(0) * 2 - 1
        frames = [np.array(f).astype(np.uint8) for f in frames]

        masks = self.read_mask(mask_path, size)
        binary_masks = [
            np.expand_dims((np.array(m) != 0).astype(np.uint8), 2) for m in masks
        ]
        masks = to_tensors()(masks).unsqueeze(0)
        imgs, masks = imgs.to(self.device), masks.to(self.device)
        comp_frames = [None] * video_length

        # completing holes by e2fgvi
        print(f'Start test...')
        import pdb;pdb.set_trace()
        for f in tqdm(range(0, video_length, self.neighbor_stride)):
            # import pdb;pdb.set_trace()
            neighbor_ids = [
                i for i in range(max(0, f - self.neighbor_stride),
                                min(video_length, f + self.neighbor_stride + 1))
            ]
            ref_ids = self.get_ref_index(f, neighbor_ids, video_length)
            selected_imgs = imgs[:1, neighbor_ids + ref_ids, :, :, :]
            selected_masks = masks[:1, neighbor_ids + ref_ids, :, :, :]
            with torch.no_grad():
                masked_imgs = selected_imgs * (1 - selected_masks)
                mod_size_h = 60
                mod_size_w = 108
                h_pad = (mod_size_h - h % mod_size_h) % mod_size_h
                w_pad = (mod_size_w - w % mod_size_w) % mod_size_w
                masked_imgs = torch.cat(
                    [masked_imgs, torch.flip(masked_imgs, [3])],
                    3)[:, :, :, :h + h_pad, :]
                masked_imgs = torch.cat(
                    [masked_imgs, torch.flip(masked_imgs, [4])],
                    4)[:, :, :, :, :w + w_pad]
                pred_imgs, _ = self.model(masked_imgs, len(neighbor_ids))
                pred_imgs = pred_imgs[:, :, :h, :w]
                pred_imgs = (pred_imgs + 1) / 2
                pred_imgs = pred_imgs.cpu().permute(0, 2, 3, 1).numpy() * 255
                for i in range(len(neighbor_ids)):
                    idx = neighbor_ids[i]
                    img = np.array(pred_imgs[i]).astype(
                        np.uint8) * binary_masks[idx] + frames[idx] * (
                            1 - binary_masks[idx])
                    if comp_frames[idx] is None:
                        comp_frames[idx] = img
                    else:
                        comp_frames[idx] = comp_frames[idx].astype(
                            np.float32) * 0.5 + img.astype(np.float32) * 0.5

        # saving videos
        print('Saving videos...')
        save_dir_name = f"./demo/inpaint_res"
        ext_name = '_results.mp4'
        save_base_name = video_path.split('/')[-2]
        save_name = save_base_name + ext_name
        if not os.path.exists(save_dir_name):
            os.makedirs(save_dir_name)
        save_path = os.path.join(save_dir_name, save_name)
        writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                self.default_fps, size)
        for f in range(video_length):
            comp = comp_frames[f].astype(np.uint8)
            writer.write(cv2.cvtColor(comp, cv2.COLOR_BGR2RGB))
        writer.release()
        print(f'Finish test! The result video is saved in: {save_path}.')



        
# if __name__ == '__main__':
    # inpainter = Inpainter()
    # inpainter.main_worker(video_path = "/data02/lxd/py_project/open_gpt/videogpt/demo/rvos_res/images/0c04834d61/", mask_path = "/data02/lxd/py_project/open_gpt/videogpt/demo/rvos_res/results/0c04834d61/")