import pdb
import os
import cv2
from PIL import Image
import math
import numpy as np
import datetime
import torch
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

from moviepy.editor import VideoFileClip
from google.cloud import vision
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def has_audio(video_path):
    try:
        video_clip = VideoFileClip(video_path)
        audio = video_clip.audio
        return audio is not None
    except Exception as e:
        print(f"Error: {e}")
        return False


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


class TemporalBase(object):
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

        ####ocr initial
        if self.config.google_cloud.CLOUD_VISION_API_KEY is None:
            print("### Please set the GOOGLE_CLOUD_API_KEY environment variable.")
            self.client = None
        else:
            self.client = vision.ImageAnnotatorClient(
                client_options={
                    "api_key": self.config.google_cloud.CLOUD_VISION_API_KEY,
                    "quota_project_id": self.config.google_cloud.QUOTA_PROJECT_ID,
                }
            )

        self.llm = OpenAI(
            openai_api_key = self.config.openai.GPT_API_KEY, 
            model_name = self.config.openai.AGENT_GPT_MODEL_NAME, 
            base_url=self.config.openai.PROXY,
            temperature = 0
        )

        ####audio initial
        # self.whisper_model = whisper.load_model("base")
        self.whisper_model = whisper.load_model("large").to("cuda")

        ####other args
        self.frames = None
        self.fps = None
        self.video_len = None
        self.sql_path = None
        self.step = None
        self.sub_frames = None

    def inital_video(self, video_path, step=30, start_time=0, end_time=None):
        cap = cv2.VideoCapture(video_path)
        self.fps = cap.get(cv2.CAP_PROP_FPS)

        self.step = step

        count = 0
        self.frames = []
        if end_time is None:
            end_time = cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.fps

        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                count += 1
                if count > (start_time * self.fps) and count <= (end_time * self.fps):
                    self.frames.append(frame)

                if count == (end_time * self.fps):
                    break
            else:
                break

        self.video_len = len(self.frames)
        self.sub_frames = self.frames[:: self.step]

    def reset_video(self):
        self.frames = None
        self.fps = None
        self.video_len = None
        self.sql_path = None
        self.step = None
        self.sub_frames = None

    def build_database(self, video_path):
        video_dir = os.path.dirname(video_path)
        video_name = os.path.basename(video_path).split(".")[0]
        self.sql_path = os.path.join(video_dir, video_name + ".db")

        conn = sqlite3.connect(self.sql_path)
        cursor = conn.cursor()

        cursor.execute(
            "CREATE TABLE IF NOT EXISTS temporaldb (frame_id INTEGER PRIMARY KEY, frame_time TEXT)"
        )
        conn.commit()

        cursor.execute("ALTER TABLE temporaldb ADD visual_content TEXT")
        conn.commit()

        cursor.execute("ALTER TABLE temporaldb ADD subtitles TEXT")
        conn.commit()

        cursor.execute("ALTER TABLE temporaldb ADD audio_content TEXT")
        conn.commit()

        cursor.execute("PRAGMA table_info(temporaldb)")
        rows = cursor.fetchall()
        print("### Table temporaldb now is", rows)
        conn.commit()

        for id_i in range(0, len(self.sub_frames)):
            frame_id_i = id_i * self.step
            id_time_second = format_seconds_to_time(math.floor(frame_id_i / (self.fps)))
            cursor.execute(
                "INSERT INTO temporaldb (frame_id, frame_time) VALUES (?, ?)",
                (frame_id_i, id_time_second),
            )
            conn.commit()

        conn.close()

    def run_on_video(self, video_path, step=30):
        self.inital_video(video_path, step)

        video_dir = os.path.dirname(video_path)
        video_name = os.path.basename(video_path).split(".")[0]
        self.sql_path = os.path.join(video_dir, video_name + ".db")
        conn = sqlite3.connect(self.sql_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='temporaldb';"
        )
        rows = cursor.fetchall()
        if len(rows) == 0:
            self.build_database(video_path)
            self.run_VideoCaption()
            self.run_OpticalCharacterRecognition()
            self.run_AudioToText(video_path)

        ####visual
        conn = sqlite3.connect(self.sql_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM temporaldb")
        rows = cursor.fetchall()
        print("### Table temporaldb now is", rows)
        conn.close()

        return rows[0]

    def run_on_question(self, question):
        ####visual
        conn = sqlite3.connect(self.sql_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM temporaldb")
        rows = cursor.fetchall()
        print("### Table temporaldb now is", rows)
        conn.close()

        db = SQLDatabase.from_uri("sqlite:///" + self.sql_path)
        db_chain = SQLDatabaseChain(llm=self.llm, database=db, top_k=100, verbose=True)
        result = db_chain.run(question)

        return result

    def run_VideoCaption(self):
        conn = sqlite3.connect(self.sql_path)
        cursor = conn.cursor()

        for id_i in range(0, len(self.sub_frames)):
            frame_id_i = id_i * self.step
            caption_text = self.caption_by_img(self.sub_frames[id_i])
            cursor.execute(
                "UPDATE temporaldb SET visual_content = ? WHERE frame_id = ?;",
                (caption_text, frame_id_i),
            )
            conn.commit()

        conn.close()

    def run_OpticalCharacterRecognition(self):
        conn = sqlite3.connect(self.sql_path)
        cursor = conn.cursor()

        for id_i in range(0, len(self.sub_frames)):
            frame_id_i = id_i * self.step
            ocr_text, _ = self.detect_text_by_img(self.sub_frames[id_i])
            if ocr_text is None:
                continue
            else:
                cursor.execute(
                    "UPDATE temporaldb SET subtitles = ? WHERE frame_id = ?;",
                    (ocr_text, frame_id_i),
                )
                conn.commit()

        conn.close()

    def run_AudioToText(self, video_path):
        conn = sqlite3.connect(self.sql_path)
        cursor = conn.cursor()

        if not has_audio(video_path):
            print("### No audio")
        else:
            _, audio_segments = self.audio_to_text_by_path(video_path)

            for seg_i in audio_segments:
                start_frame = math.floor(seg_i["start"] * self.fps)
                end_frame = math.floor(seg_i["end"] * self.fps)

                cursor.execute(
                    "SELECT * FROM temporaldb WHERE frame_id >= ? AND frame_id < ?;",
                    (start_frame, end_frame),
                )

                # Fetch the row that matches the time condition
                rows = cursor.fetchall()

                if rows:
                    for row in rows:
                        # If a matching row is found, update its value to 10
                        row_id = row[
                            0
                        ]  # Assuming the first column is the ID column (replace with your actual column index)
                        new_value = seg_i["text"]

                        # SQL query to update the value for the matching row
                        cursor.execute(
                            "UPDATE temporaldb SET audio_content = ? WHERE frame_id = ?;",
                            (new_value, row_id),
                        )
                        conn.commit()
                else:
                    print("### No row found that matches the time condition.")

            conn.close()

    def detect_text_by_img(self, image):
        """Detects text in the file."""
        if self.client is None:
            return None, None
        else:
            success, content = cv2.imencode(".jpg", image)
            content = content.tobytes()

            image = vision.Image(content=content)
            response = self.client.text_detection(image=image)
            texts = response.text_annotations

            if len(texts) == 0:
                return None, None
            else:
                descriptions = texts[0].description
                vertices = [
                    f"({vertex.x},{vertex.y})"
                    for vertex in texts[0].bounding_poly.vertices
                ]

                if response.error.message:
                    raise Exception(
                        "{}\nFor more info on error messages, check: "
                        "https://cloud.google.com/apis/design/errors".format(
                            response.error.message))
            return descriptions, vertices

    def caption_by_img(self, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        raw_image = Image.fromarray(rgb_image)
        inputs = self.processor(raw_image, return_tensors="pt").to(
            self.device, self.torch_dtype
        )
        out = self.model.generate(**inputs)
        answer = self.processor.decode(out[0], skip_special_tokens=True)
        return answer

    def audio_to_text_by_path(self, video_path):
        audio = self.whisper_model.transcribe(video_path,language="english")
        text = audio["text"]
        audio_text_with_time = [
            {"start": seg["start"], "end": seg["end"], "text": seg["text"]}
            for seg in audio["segments"]
        ]

        return text, audio_text_with_time
