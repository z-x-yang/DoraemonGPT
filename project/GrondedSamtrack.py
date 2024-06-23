import importlib
import sys
import os
sys.path.append(f"./project/aot-benchmark")
sys.path.append(f"./project/aot-benchmark/tools")

import cv2
from PIL import Image
from skimage.morphology.binary import binary_dilation

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from networks.models import build_vos_model
from networks.engines import build_engine
from utils.checkpoint import load_network

from dataloaders.eval_datasets import VOSTest
import dataloaders.video_transforms as tr
from utils.image import save_mask


import json,cv2,os
import ipdb
import cv2
import numpy as np
import supervision as sv
import argparse
import torch
import torchvision
import sys
sys.path.append(f"./project/Grounded-Segment-Anything")
# sys.path.append
from groundingdino.util.inference import Model
from segment_anything import SamPredictor,sam_model_registry
sys.path.append(f"./project/Grounded-Segment-Anything/EfficientSAM")
from MobileSAM.setup_mobile_sam import setup_model

import cv2




class GrandedSamTracker(object):
    def __init__(
        self,
        device="cuda:0",
    ):
        self.device = device
        if device.startswith("cuda"):
            self.gpu_id = int(device.split(":")[-1])
        else:
            self.gpu_id = 0
        engine_config = importlib.import_module('configs.' + 'pre_ytb_dav')
        cfg = engine_config.EngineConfig('default', 'r50_deaotl')
        cfg.TEST_CKPT_PATH = f"./checkpoints/AOT/R50_DeAOTL_PRE_YTB_DAV.pth"
        cfg.TEST_MIN_SIZE = None
        cfg.TEST_MAX_SIZE = 480 * 1.3 * 800. / 480.
        video_fps = 15

        # Load pre-trained model
        # print('Build AOT model.')
        model = build_vos_model(cfg.MODEL_VOS, cfg).to(device=self.device)

        # print('Load checkpoint from {}'.format(cfg.TEST_CKPT_PATH))
        model, _ = load_network(model, cfg.TEST_CKPT_PATH, self.gpu_id)
        

        # print('Build AOT engine.')
        engine = build_engine(cfg.MODEL_ENGINE,
                                phase='eval',
                                aot_model=model,
                                gpu_id=self.gpu_id,
                                long_term_mem_gap=cfg.TEST_LONG_TERM_MEM_GAP)

        # Prepare datasets for each sequence
        self.transform = transforms.Compose([
            tr.MultiRestrictSize(cfg.TEST_MIN_SIZE, cfg.TEST_MAX_SIZE,
                                    cfg.TEST_FLIP, cfg.TEST_MULTISCALE,
                                    cfg.MODEL_ALIGN_CORNERS),
            tr.MultiToTensor()
        ])

        # Building GroundingDINO inference model
        # grounding_dino_model = Model(model_config_path=f"./project/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", model_checkpoint_path=f"./checkpoints/GroundedSAM/groundingdino_swint_ogc.pth")
        grounding_dino_model = Model(model_config_path=f"./project/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinB.py", model_checkpoint_path=f"./checkpoints/GroundedSAM/groundingdino_swinb_cogcoor.pth")
        # # Building MobileSAM predictor
        checkpoint = torch.load(f"./checkpoints/GroundedSAM/mobile_sam.pt")
        mobile_sam = setup_model()
        mobile_sam.load_state_dict(checkpoint, strict=True)
        mobile_sam.to(device=self.device)
        sam_predictor = SamPredictor(mobile_sam)
        # ######
        self.model = model
        self.engine = engine
        self.grounding_dino_model = grounding_dino_model
        self.sam_predictor = sam_predictor
        self.cfg = cfg
        self.output_folder_root = f"./demo/rvos_res/"

    def extract_frames(self, output_folder, video_path):
        video_capture = cv2.VideoCapture(video_path)

        frames = []
        i = 0
        while video_capture.isOpened():
            ret, frame = video_capture.read()

            if not ret:
                break
            
            os.makedirs(output_folder, exist_ok=True)
            output_path = os.path.join(output_folder, str(i).zfill(5) + ".jpg")
            cv2.imwrite(output_path, frame)

            frames.append(frame)
            i = i + 1

        video_capture.release()
        return frames



    
    def get_max_mask(self, exp, frame_list):
        flag_use_max=True
        if flag_use_max:
            max_mask = None
            max_frame_id = None
            max_cfd = 0
            for frame_i in range(len(frame_list)):
                image = frame_list[frame_i]

                # detect objects
                detections = self.grounding_dino_model.predict_with_classes(
                    image=image,
                    classes=exp,
                    box_threshold=0.1,
                    text_threshold=0.1
                )
                # NMS post process
                # print(f"Before NMS: {len(detections.xyxy)} boxes")
                nms_idx = torchvision.ops.nms(
                    torch.from_numpy(detections.xyxy), 
                    torch.from_numpy(detections.confidence), 
                    0.5,
                ).numpy().tolist()

                detections.xyxy = detections.xyxy[nms_idx]
                detections.confidence = detections.confidence[nms_idx]
                detections.class_id = detections.class_id[nms_idx]

                # print(f"After NMS: {len(detections.xyxy)} boxes")  

                self.sam_predictor.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                result_masks = []
                for box in detections.xyxy:
                    masks, scores, logits = self.sam_predictor.predict(
                        box=box,
                        multimask_output=True
                    )
                    index = np.argmax(scores)
                    result_masks.append(masks[index])
                detections.mask = np.array(result_masks)


                for i in range(len(detections)):
                    mask = detections.mask[i]
                    confidence = detections.confidence[i]
                    class_id = detections.class_id[i]
                    if confidence > max_cfd:
                        max_cfd = confidence
                        max_mask = mask
                        max_frame_id = frame_i 
            
            if max_cfd == 0:
                return {"max_mask":None , "max_cfd":None, "max_frame_id":None}
            
            max_mask = max_mask.astype(np.uint8)
            return {"max_mask":max_mask , "max_cfd":max_cfd, "max_frame_id":max_frame_id}
        else:
            max_mask = None
            max_frame_id = None
            max_cfd = 0
            frame_i = 0

            image = frame_list[frame_i]

            # detect objects
            detections = self.grounding_dino_model.predict_with_classes(
                image=image,
                classes=exp,
                box_threshold=0.1,
                text_threshold=0.1
            )
            # NMS post process
            # print(f"Before NMS: {len(detections.xyxy)} boxes")
            nms_idx = torchvision.ops.nms(
                torch.from_numpy(detections.xyxy), 
                torch.from_numpy(detections.confidence), 
                0.5,
            ).numpy().tolist()

            detections.xyxy = detections.xyxy[nms_idx]
            detections.confidence = detections.confidence[nms_idx]
            detections.class_id = detections.class_id[nms_idx]

            # print(f"After NMS: {len(detections.xyxy)} boxes")  

            self.sam_predictor.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            result_masks = []
            for box in detections.xyxy:
                masks, scores, logits = self.sam_predictor.predict(
                    box=box,
                    multimask_output=True
                )
                index = np.argmax(scores)
                result_masks.append(masks[index])
            detections.mask = np.array(result_masks)

            for i in range(len(detections)):
                mask = detections.mask[i]
                confidence = detections.confidence[i]
                class_id = detections.class_id[i]
                if confidence > max_cfd:
                    max_cfd = confidence
                    max_mask = mask
                    max_frame_id = frame_i 
        
            max_mask = max_mask.astype(np.uint8)
            return {"max_mask":max_mask , "max_cfd":max_cfd, "max_frame_id":max_frame_id}


    def get_tracking_res(self, exp, video_path):

        seq_name = video_path.split("/")[-1].split(".")[0]

        output_folder_images = self.output_folder_root + "images"+ "/"
        os.makedirs(os.path.join(output_folder_images, seq_name), exist_ok=True)
        output_folder_mask = self.output_folder_root  + "mask"+ "/" 
        os.makedirs(os.path.join(output_folder_mask, seq_name), exist_ok=True)
        output_folder_result =  self.output_folder_root + "results"+ "/"
        os.makedirs(os.path.join(output_folder_result, seq_name), exist_ok=True)

        frame_list = self.extract_frames(os.path.join(output_folder_images, seq_name), video_path)
        meta_max_info = self.get_max_mask(exp, frame_list)
        cv2.imwrite(os.path.join(output_folder_mask, seq_name ,str(meta_max_info['max_frame_id']).zfill(5)+".png"), meta_max_info['max_mask']*255)
        cv2.imwrite(os.path.join(output_folder_result, seq_name ,str(meta_max_info['max_frame_id']).zfill(5)+".png"), meta_max_info['max_mask']*255)


        image_root = output_folder_images
        label_root = output_folder_mask

        # print('Build a dataset for sequence {}.'.format(seq_name))
        seq_images = np.sort(os.listdir(os.path.join(image_root, seq_name)))
        index = list(seq_images).index(str(meta_max_info['max_frame_id']).zfill(5)+'.jpg')
        list1 = seq_images[:index+1][::-1]
        list2 = seq_images[index:]

        for flag in [True,False]:
            if flag:
                seq_images = list1
            else:
                seq_images = list2
                
            if len(seq_images) == 1:
                continue
            else:
                seq_labels = [seq_images[0].replace('jpg', 'png')]
                seq_dataset = VOSTest(image_root,os.path.join(label_root,seq_name),seq_name,seq_images,seq_labels,transform=self.transform)


                # Infer
                output_root =  self.output_folder_root + "results"+ "/"
                output_mask_seq_root = os.path.join(output_root, seq_name)
                os.makedirs(output_mask_seq_root, exist_ok=True)
                # print('Build a dataloader for sequence {}.'.format(seq_name))
                seq_dataloader = DataLoader(seq_dataset,
                                            batch_size=1,
                                            shuffle=False,
                                            num_workers=self.cfg.TEST_WORKERS,
                                            pin_memory=True)


                # print('Start the inference of sequence {}:'.format(seq_name))
                self.model.eval()
                self.engine.restart_engine()
                with torch.no_grad():
                    for frame_idx, samples in enumerate(seq_dataloader):
                    
                        sample = samples[0]
                        img_name = sample['meta']['current_name'][0]

                        obj_nums = sample['meta']['obj_num']
                        output_height = sample['meta']['height']
                        output_width = sample['meta']['width']
                        obj_idx = sample['meta']['obj_idx']

                        obj_nums = [int(obj_num) for obj_num in obj_nums]
                        obj_idx = [int(_obj_idx) for _obj_idx in obj_idx]

                        current_img = sample['current_img']
                        current_img = current_img.cuda(self.gpu_id, non_blocking=True)

                        if frame_idx == 0:
                            current_label = sample['current_label'].cuda(
                                self.gpu_id, non_blocking=True).float()
                            current_label = F.interpolate(current_label,
                                                            size=current_img.size()[2:],
                                                            mode="nearest")
                            # add reference frame
                            self.engine.add_reference_frame(current_img,
                                                        current_label,
                                                        frame_step=0,
                                                        obj_nums=obj_nums)
                        else:
                            # print('Processing image {}...'.format(img_name))
                            # predict segmentation
                            self.engine.match_propogate_one_frame(current_img)
                            pred_logit = self.engine.decode_current_logits(
                                (output_height, output_width))
                            pred_prob = torch.softmax(pred_logit, dim=1)
                            pred_label = torch.argmax(pred_prob, dim=1,
                                                        keepdim=True).float()
                            _pred_label = F.interpolate(pred_label,
                                                        size=self.engine.input_size_2d,
                                                        mode="nearest")
                            # update memory
                            self.engine.update_memory(_pred_label)

                            # save results
                            output_mask_path = os.path.join(
                                output_mask_seq_root,
                                img_name.split('.')[0] + '.png')
                            pred_label = Image.fromarray(
                                pred_label.mul(255).squeeze(0).squeeze(0).cpu().numpy().astype(
                                    'uint8')).convert('L')
                            pred_label.save(output_mask_path)




if __name__ == "__main__":
    tracker = GrandedSamTracker()

    exp = "the car"
    video_path = "/data02/lxd/py_project/open_gpt/videogpt/demo/nextqa_demo/0c04834d61.mp4"
 
    tracker.get_tracking_res(exp,video_path )