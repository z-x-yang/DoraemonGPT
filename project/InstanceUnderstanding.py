import pdb
import os
import cv2
from PIL import Image
import math
import numpy as np
import datetime
import re
import torch
import ffmpeg
import sqlite3
import whisper
import openai
from torch.cuda.amp import autocast as autocast
from transformers import (
    pipeline,
    BlipProcessor,
    BlipForConditionalGeneration,
    BlipForQuestionAnswering,
)
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.llms.openai import OpenAI
from langchain_experimental.sql import SQLDatabaseChain
from langchain import OpenAI, SQLDatabase

import sys

sys.path.append("./project")
sys.path.append("./project/Grounded-Segment-Anything")
from MultiTracking import detect_by_path
import torchvision.transforms as transforms
from load_internvideo import *
import decord
from decord import VideoReader
from decord import cpu
from project.GrondedSamtrack import GrandedSamTracker


def format_seconds_to_time(seconds):
    # Convert seconds to a timedelta object
    time_delta = datetime.timedelta(seconds=seconds)

    # Extract the hours, minutes, and seconds components from the timedelta
    hours = time_delta.seconds // 3600
    minutes = (time_delta.seconds // 60) % 60
    seconds = time_delta.seconds % 60

    # Use string formatting to create the formatted time string
    formatted_time = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    return formatted_time


def get_files(path):
    files = [p for p in path.glob("*") if p.is_file()]
    dirs = [p for p in path.glob("*/") if p.is_dir()]
    for d in dirs:
        files.extend(get_files(d))
    return files


class InstanceBase(object):
    def __init__(
        self,
        device="cpu",
        config = None,
    ):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        # ####caption initial
        self.config = config
        self.device = device
        self.torch_dtype = torch.float16 if "cuda" in device else torch.float32
        self.processor = BlipProcessor.from_pretrained("./checkpoints/blip")
        self.model = BlipForConditionalGeneration.from_pretrained(
            "./checkpoints/blip", torch_dtype=self.torch_dtype
        ).to(self.device)

        self.llm = OpenAI(
            openai_api_key = self.config.openai.GPT_API_KEY, 
            model_name = self.config.openai.AGENT_GPT_MODEL_NAME, 
            base_url=self.config.openai.PROXY,
            temperature = 0
        )
        self.tracker = GrandedSamTracker()

        self.fps = None
        self.video_path = None
        self.sql_path = None
        self.step = None

        self.pro_dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.checkpoint_dir = self.pro_dir_path + "/" + "checkpoints"
        self.save_dir = self.pro_dir_path + "/" + "demo" + "/" + "track_res"
        self.res_dir = None

    def reset_video(self):
        self.frames = None
        self.fps = None
        self.video_len = None
        self.sql_path = None
        self.step = None
        self.sub_frames = None

    def inital_database(self):
        res_dir_labels = self.res_dir / "labels"
        sorted_paths = sorted(res_dir_labels.iterdir(), key=lambda p: p.name)
        res_label_txt = sorted_paths[0]
        coco_txt = self.checkpoint_dir + "/" + "coco.txt"

        coco_list = []
        with open(coco_txt, "r") as f:
            for line in f.readlines():
                cat_i = line.strip()
                coco_list.append(cat_i)

        result_list = []
        with open(res_label_txt, "r") as f:
            for line in f.readlines():
                dict_i = {}
                value_i = line.strip().split()
                dict_i["frame_idx"] = int(value_i[0]) * self.step - 1
                dict_i["time_idx"] = format_seconds_to_time(
                    math.floor(dict_i["frame_idx"] / (self.fps))
                )
                dict_i["id"] = value_i[1]
                dict_i["category"] = value_i[-2]
                dict_i["position"] = list(map(int, value_i[2:6]))
                result_list.append(dict_i)

        id_set = set()
        for d in result_list:
            id_set = id_set.union(set(d.get("id")))

        video_dir = os.path.dirname(self.video_path)
        video_name = os.path.basename(self.video_path).split(".")[0]
        self.sql_path = os.path.join(video_dir, video_name + ".db")

        conn = sqlite3.connect(self.sql_path)
        cursor = conn.cursor()

        cursor.execute(
            "CREATE TABLE IF NOT EXISTS instancedb (obj_id INTEGER PRIMARY KEY)"
        )
        conn.commit()

        cursor.execute("ALTER TABLE instancedb ADD category TEXT")
        conn.commit()

        cursor.execute("ALTER TABLE instancedb ADD identification TEXT")
        conn.commit()

        cursor.execute("ALTER TABLE instancedb ADD trajectory TEXT")
        conn.commit()

        cursor.execute("ALTER TABLE instancedb ADD action TEXT")
        conn.commit()

        for id_i in id_set:
            result_id = [d for d in result_list if d.get("id") == id_i]
            if len(result_id) == 0:
                continue

            result_id = sorted(result_id, key=lambda x: x["frame_idx"])

            res_dir_crops = self.res_dir / "crops" / result_id[0]["category"] / id_i

            sorted_paths = sorted(
                os.listdir(res_dir_crops),
                key=lambda x: int(re.search(r"_(\d+)\.jpg", x).group(1)),
            )

            image_id_i = cv2.imread(str(res_dir_crops / sorted_paths[0]))

            identification_i = self.caption_by_img(image_id_i)

            motion_p = ""
            for res_idx in result_id:
                motion_p += (
                    "At "
                    + str(res_idx["time_idx"])
                    + ", "
                    + str(res_idx["position"])
                    + "; "
                )

            motion = self.motion_inference(res_dir_crops)
            cursor.execute(
                "INSERT INTO instancedb (obj_id, category, identification, trajectory, action) VALUES (?, ?, ?, ?, ?)",
                (
                    id_i,
                    coco_list[int(result_id[0]["category"])],
                    identification_i,
                    motion_p,
                    motion,
                ),
            )
            conn.commit()

    def caption_by_img(self, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        raw_image = Image.fromarray(rgb_image)
        inputs = self.processor(raw_image, return_tensors="pt").to(
            self.device, self.torch_dtype
        )
        out = self.model.generate(**inputs)
        answer = self.processor.decode(out[0], skip_special_tokens=True)

        return answer

    def img_to_video(self, image_folder):
        image_files = [
            os.path.join(image_folder, f)
            for f in os.listdir(image_folder)
            if f.endswith(".jpg")
        ]
        image_files.sort()
        first_image = cv2.imread(image_files[0])
        h, w = first_image.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            os.path.join(image_folder, "output.mp4"),
            fourcc,
            math.ceil(self.fps / self.step),
            (w, h),
        )
        for image_file in image_files:
            image = cv2.imread(image_file, cv2.IMREAD_COLOR)
            if image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            image = cv2.resize(image, (w, h))
            out.write(image)
        out.release()
        return os.path.join(image_folder, "output.mp4")

    def loadvideo_decord_origin(
        self,
        sample,
        sample_rate_scale=1,
        new_width=384,
        new_height=384,
        clip_len=8,
        frame_sample_rate=2,
        num_segment=1,
    ):
        # sample = self.img_to_video(sample)

        # fname = sample
        # vr = VideoReader(fname, num_threads=1, ctx=cpu(0))
        # # handle temporal segments
        # converted_len = int(clip_len * frame_sample_rate)
        # seg_len = len(vr) // num_segment
        # duration = max(len(vr) // vr.get_avg_fps(), 8)

        # all_index = []
        # for i in range(num_segment):
        #     index = np.linspace(0, seg_len, num=int(duration))
        #     index = np.clip(index, 0, seg_len - 1).astype(np.int64)
        #     index = index + i * seg_len
        #     all_index.extend(list(index))
        # all_index = all_index[:: int(sample_rate_scale)]
        # vr.seek(0)
        # buffer = vr.get_batch(all_index).asnumpy()
        # return buffer
        image_folder = sample
        image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(".jpg") or f.endswith(".png")]
        image_files.sort()
        first_image = cv2.imread(image_files[0], cv2.IMREAD_COLOR)
        h, w = first_image.shape[:2]

        frame_sample_rate = frame_sample_rate
        converted_len = int(clip_len * frame_sample_rate)
        seg_len = len(image_files) // num_segment
        duration = max(len(image_files) / frame_sample_rate, 8)

        all_index = []
        for i in range(num_segment):
            index = np.linspace(0, seg_len, num=int(duration))
            index = np.clip(index, 0, seg_len - 1).astype(np.int64)
            index = index + i * seg_len
            all_index.extend(list(index))
        all_index = all_index[:: int(sample_rate_scale)]

        buffer = []
        for idx in all_index:
            image = cv2.imread(image_files[int(idx)], cv2.IMREAD_COLOR)
            if image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            image = cv2.resize(image, (new_width, new_height))
            buffer.append(image)

        return np.array(buffer)

    def motion_inference(self, video_path):
        intern_action = load_intern_action(self.device)
        data = self.loadvideo_decord_origin(sample=video_path)
        topil = T.ToPILImage()
        trans_action = transform_action()
        # InternVideo
        action_index = np.linspace(0, len(data) - 1, 8).astype(int)
        tmp, tmpa = [], []
        image_size = 384
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                normalize,
            ]
        )

        for i, img in enumerate(data):
            tmp.append(transform(img).to(self.device).unsqueeze(0))
            if i in action_index:
                tmpa.append(topil(img))
        action_tensor = trans_action(tmpa)
        TC, H, W = action_tensor.shape
        action_tensor = (
            action_tensor.reshape(1, TC // 3, 3, H, W)
            .permute(0, 2, 1, 3, 4)
            .to(self.device)
        )
        with torch.no_grad():
            prediction = intern_action(action_tensor)
            prediction = F.softmax(prediction, dim=1).flatten()
            prediction = kinetics_classnames[str(int(prediction.argmax()))]
            return prediction

    def ref_vos(self, video_path, question):
        flag_ref_vos = self.llm(f"Please determine if this task is related to inpaint, generally referring to the word inpaint in the task. Reply 0 if it is related and 1 if it is not. The task is as follows:{question}.")
        if int(flag_ref_vos) == 0:
            try:
                exp = self.llm(f"Extract the most important part of the subject from the sentence {question}, returning only one phrase, which is required to be a noun, or a noun with an attributive.")
                print(f"### Start rvos, ###exp:{exp}")
                self.tracker.get_tracking_res(exp,video_path )
            except:
                print(f"### Error in rvos.")
        else:
            print(f"### No need for rvos.")
        # conn = sqlite3.connect(self.sql_path)
        # cursor = conn.cursor()
        # cursor.execute("INSERT INTO instancedb (obj_id, category, identification, trajectory, action) VALUES (?, ?, ?, ?, ?)",(-1,exp,exp,None,None,),)
        # conn.commit()

    def run_on_video(self, video_path, question, step=10):
        self.video_path = video_path
        cap = cv2.VideoCapture(self.video_path)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.step = step
        self.res_dir = detect_by_path(
            self.checkpoint_dir,
            self.video_path,
            self.save_dir,
            device=self.device,
            step=step,
        )

        video_dir = os.path.dirname(video_path)
        video_name = os.path.basename(video_path).split(".")[0]
        self.sql_path = os.path.join(video_dir, video_name + ".db")
        conn = sqlite3.connect(self.sql_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='instancedb';"
        )
        rows = cursor.fetchall()

        if len(rows) == 0:
            self.inital_database()
            self.ref_vos(video_path,question)

        ####visual
        conn = sqlite3.connect(self.sql_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM instancedb")
        rows = cursor.fetchall()
        print("### Table instancedb now is", rows)
        conn.close()

        return rows[0]

    def run_on_question(self, question):
        ####visual
        conn = sqlite3.connect(self.sql_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM instancedb")
        rows = cursor.fetchall()
        print("### Table instancedb now is", rows)
        conn.close()

        db = SQLDatabase.from_uri("sqlite:///" + self.sql_path)
        db_chain = SQLDatabaseChain(llm=self.llm, database=db, verbose=True)
        result = db_chain.run(question)

        return result
