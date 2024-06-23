# Licensed under the MIT License.
# -*- coding: utf-8 -*-

import os
import sys
sys.path.append(f"./")
sys.path.append(f"../")

import random, math, cv2, inspect, tempfile, csv
import torch
from PIL import Image, ImageDraw, ImageOps, ImageFont
import numpy as np
import argparse
from omegaconf import OmegaConf

import openai
import langchain
import sqlite3
# langchain.debug = True
from langchain.prompts import PromptTemplate
from langchain.agents.initialize import initialize_agent
from langchain.agents.tools import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.llms.openai import OpenAI

from project.TemporalUnderstanding import TemporalBase
from project.InstanceUnderstanding import InstanceBase
from project.ExampleSelector import CustomExampleSelector
from project.TreeSearch import ReThinking
from project.sql_template import (
    _sqlite_prompt,
    COUNTING_EXAMPLE_PROMPT,
    PROMPT_SUFFIX,
    TEMPORAL_EXAMPLE_PROMPT,
    REASONFINDER_ADDITION_PROMPT,
    HOWSEEKER_ADDITION_PROMPT,
    DESCRIP_EXAMPLE_PROMPT,
    DESCRIP_ADDITION_PROMPT,
)
from project.E2FGVI.Inpainter import Inpainter

from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.llms.openai import OpenAI
from langchain_experimental.sql import SQLDatabaseChain
from langchain import OpenAI, SQLDatabase




def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def prompts(name, description):
    
    def decorator(func):
        func.name = name
        func.description = description
        return func

    return decorator


class TemporalTool:
    def __init__(self, device, config):
        self.device = device
        self.config = config
        self.llm = OpenAI(
            openai_api_key = self.config.openai.GPT_API_KEY,
            model_name= self.config.openai.GPT_MODEL_NAME,
            temperature = 0,
            base_url=self.config.openai.PROXY
        )
        self.sql_prompt = PromptTemplate(
            input_variables = ["input", "table_info", "top_k"],
            template = _sqlite_prompt + TEMPORAL_EXAMPLE_PROMPT + PROMPT_SUFFIX,
        )

    @prompts(
        name = "TemporalTool",
        description = "Useful when you need to process temporal information in videos."
        "The input to this tool must be a string for the video path, and a string for the question. Concatenate them using # as the separator."
        "For example: the input is ./videos/xxx.mp4#What is he talking about when a girl is playing violin? ",
    )
    def inference(self, input):
        if "#" in input:
            tmp = input.split("#")
            if len(tmp) == 2:
                video_path = tmp[0]
                question = tmp[1]
            else: 
                result = "There is an error in the input of the tool, please check the input and try again."
                return result
        else:
            result = "There is an error in the input of the tool, please check the input and try again."
            return result
        
        video_dir = os.path.dirname(video_path)
        video_name = os.path.basename(video_path).split(".")[0]
        self.sql_path = os.path.join(video_dir, video_name + ".db")

        db = SQLDatabase.from_uri(
            "sqlite:///" + self.sql_path, sample_rows_in_table_info=2
        )
        db_chain = SQLDatabaseChain(
            llm=self.llm, database=db, top_k=self.config.tool.TOP_K, verbose=True, prompt=self.sql_prompt
        )

        try:
            result = db_chain.run(question)
        except:
            result ="There is an error. Try to ask the question in a different way."

        print(
            f"\nProcessed TemporalTool, Input Video: {video_path}, Input Question: {question}, "
            f"Output Answer: {result}"
        )
        return result


class CountingTool:
    def __init__(self, device, config):
        self.device = device
        self.config = config
        self.llm = OpenAI(
            openai_api_key = self.config.openai.GPT_API_KEY,
            model_name = self.config.openai.GPT_MODEL_NAME,
            temperature = 0,
            base_url=self.config.openai.PROXY
        )
        self.sql_prompt = PromptTemplate(
            input_variables = ["input", "table_info", "top_k"],
            template = _sqlite_prompt + COUNTING_EXAMPLE_PROMPT+ PROMPT_SUFFIX,
        )

    @prompts(
        name = "CountingTool",
        description = "Useful when you need to count object number."
        "The input to this tool must be a string for the video path, and a string for the question. Concatenate them using # as the separator."
        "For example: the input is ./videos/xxx.mp4#How many fish are here? ",
    )
    def inference(self, input):
        if "#" in input:
            tmp = input.split("#")
            if len(tmp) == 2:
                video_path = tmp[0]
                question = tmp[1]
            else: 
                result = "There is an error in the input of the tool, please check the input and try again."
                return result
        else:
            result = "There is an error in the input of the tool, please check the input and try again."
            return result

        video_dir = os.path.dirname(video_path)
        video_name = os.path.basename(video_path).split(".")[0]
        self.sql_path = os.path.join(video_dir, video_name + ".db")

        db = SQLDatabase.from_uri(
            "sqlite:///" + self.sql_path, sample_rows_in_table_info=2
        )
        db_chain = SQLDatabaseChain(
            llm=self.llm, database=db, top_k=self.config.tool.TOP_K, verbose=True, prompt=self.sql_prompt
        )

        try:
            result = db_chain.run(question)
        except:
            result ="There is an error. Try to ask the question in a different way."

        print(
            f"\nProcessed CountingTool, Input Video: {video_path}, Input Question: {question}, "
            f"Output Answer: {result}"
        )
        return result


class ReasonFinder:
    def __init__(self, device, config):
        self.device = device
        self.config = config
        self.llm = OpenAI(
            openai_api_key = self.config.openai.GPT_API_KEY,
            model_name = self.config.openai.GPT_MODEL_NAME,
            temperature = 0,
            base_url=self.config.openai.PROXY
        )
        self.sql_prompt = PromptTemplate(
            input_variables = ["input", "table_info", "top_k"],
            template = _sqlite_prompt + REASONFINDER_ADDITION_PROMPT + PROMPT_SUFFIX,
        )

    @prompts(
        name="ReasonFinder",
        description="Useful when you need to find reasons or explanations."
        "The input to this tool must be a string for the video path, and a string for the question. Concatenate them using # as the separator."
        "For example: the input is ./videos/xxx.mp4#Why she is crying? ",
    )
    def inference(self, input):
        if "#" in input:
            tmp = input.split("#")
            if len(tmp) == 2:
                video_path = tmp[0]
                question = tmp[1]
            else: 
                result = "There is an error in the input of the tool, please check the input and try again."
                return result
        else:
            result = "There is an error in the input of the tool, please check the input and try again."
            return result

        video_dir = os.path.dirname(video_path)
        video_name = os.path.basename(video_path).split(".")[0]
        self.sql_path = os.path.join(video_dir, video_name + ".db")

        db = SQLDatabase.from_uri(
            "sqlite:///" + self.sql_path, sample_rows_in_table_info=2
        )
        db_chain = SQLDatabaseChain(
            llm=self.llm, database=db, top_k=self.config.tool.TOP_K, verbose=True, prompt=self.sql_prompt
        )

        try:
            result = db_chain.run(question)
        except:
            result ="There is an error. Try to ask the question in a different way."

        print(
            f"\nProcessed ReasonFinder, Input Video: {video_path}, Input Question: {question}, "
            f"Output Answer: {result}"
        )
        return result


class HowSeeker:
    def __init__(self, device, config):
        self.device = device
        self.config = config
        self.llm = OpenAI(
            openai_api_key = self.config.openai.GPT_API_KEY,
            model_name = self.config.openai.GPT_MODEL_NAME,
            temperature = 0,
            base_url=self.config.openai.PROXY
        )
        self.sql_prompt = PromptTemplate(
            input_variables = ["input", "table_info", "top_k"],
            template=_sqlite_prompt + HOWSEEKER_ADDITION_PROMPT + PROMPT_SUFFIX,
        )

    @prompts(
        name = "HowSeeker",
        description = "useful when you need to find methods or steps to accomplish a task."
        "The input to this tool must be a string for the video path, and a string for the question. Concatenate them using # as the separator."
        "For example: the input is ./videos/xxx.mp4#How did the children eat food? ",
    )
    def inference(self, input):
        if "#" in input:
            tmp = input.split("#")
            if len(tmp) == 2:
                video_path = tmp[0]
                question = tmp[1]
            else: 
                result = "There is an error in the input of the tool, please check the input and try again."
                return result
        else:
            result = "There is an error in the input of the tool, please check the input and try again."
            return result

        video_dir = os.path.dirname(video_path)
        video_name = os.path.basename(video_path).split(".")[0]
        self.sql_path = os.path.join(video_dir, video_name + ".db")

        db = SQLDatabase.from_uri(
            "sqlite:///" + self.sql_path, sample_rows_in_table_info=2
        )
        db_chain = SQLDatabaseChain(
            llm=self.llm, database=db, top_k=self.config.tool.TOP_K, verbose=True, prompt=self.sql_prompt
        )

        try:
            result = db_chain.run(question)
        except:
            result ="There is an error. Try to ask the question in a different way."

        print(
            f"\nProcessed HowSeeker, Input Video: {video_path}, Input Question: {question}, "
            f"Output Answer: {result}"
        )
        return result


class DescriptionTool:
    def __init__(self, device, config):
        self.device = device
        self.config = config
        self.llm = OpenAI(
            openai_api_key = self.config.openai.GPT_API_KEY,
            model_name = self.config.openai.GPT_MODEL_NAME,
            temperature = 0,
            base_url=self.config.openai.PROXY
        )
        self.sql_prompt = PromptTemplate(
            input_variables = ["input", "table_info", "top_k"],
            template=_sqlite_prompt + DESCRIP_ADDITION_PROMPT + DESCRIP_EXAMPLE_PROMPT + PROMPT_SUFFIX,
        )

    @prompts(
        name = "DescriptionTool",
        description = "Useful when you need to describe the content of a video, e.g. the audio in the video, the subtitles, the on-screen content, etc."
        "The input to this tool must be a string for the video path, and a string for the question. Concatenate them using # as the separator."
        "For example: the input is ./videos/xxx.mp4#What's in the video?",
    )
    def inference(self, input):
        if "#" in input:
            tmp = input.split("#")
            if len(tmp) == 2:
                video_path = tmp[0]
                question = tmp[1]
            else: 
                result = "There is an error in the input of the tool, please check the input and try again."
                return result
        else:
            result = "There is an error in the input of the tool, please check the input and try again."
            return result

        video_dir = os.path.dirname(video_path)
        video_name = os.path.basename(video_path).split(".")[0]
        self.sql_path = os.path.join(video_dir, video_name + ".db")

        db = SQLDatabase.from_uri(
            "sqlite:///" + self.sql_path, sample_rows_in_table_info=2
        )
        db_chain = SQLDatabaseChain(
            llm=self.llm, database=db, top_k=self.config.tool.TOP_K, verbose=True, prompt=self.sql_prompt
        )
        try:
            result = db_chain.run(question)
        except:
            result ="There is an error. Try to ask the question in a different way."

        print(
            f"\nProcessed DescriptionTool, Input Video: {video_path}, Input Question: {question}, "
            f"Output Answer: {result}"
        )
        return result


class DefaultTool:
    def __init__(self, device, config):
        self.device = device
        self.config = config
        self.llm = OpenAI(
            openai_api_key = self.config.openai.GPT_API_KEY,
            model_name = self.config.openai.GPT_MODEL_NAME,
            temperature = 0,
            base_url = self.config.openai.PROXY
        )
        self.sql_prompt = PromptTemplate(
            input_variables = ["input", "table_info", "top_k"],
            template = _sqlite_prompt + PROMPT_SUFFIX,
        )

    @prompts(
        name = "DefaultTool",
        description = "Useful when other tools can't solve the problem corresponding to the video."
        "The input to this tool must be a string for the video path, and a string for the question. Concatenate them using # as the separator."
        "For example: the input is ./videos/xxx.mp4#Are the men happy today?",
    )
    def inference(self, input):
        if "#" in input:
            tmp = input.split("#")
            if len(tmp) == 2:
                video_path = tmp[0]
                question = tmp[1]
            else: 
                result = "There is an error in the input of the tool, please check the input and try again."
                return result
        else:
            result = "There is an error in the input of the tool, please check the input and try again."
            return result

        video_dir = os.path.dirname(video_path)
        video_name = os.path.basename(video_path).split(".")[0]
        self.sql_path = os.path.join(video_dir, video_name + ".db")

        db = SQLDatabase.from_uri(
            "sqlite:///" + self.sql_path, sample_rows_in_table_info=2
        )
        db_chain = SQLDatabaseChain(
            llm=self.llm, database=db, top_k=self.config.tool.TOP_K, verbose=True, prompt=self.sql_prompt
        )
        try:
            result = db_chain.run(question)
        except:
            result ="There is an error. Try to ask the question in a different way."

        print(
            f"\nProcessed DefaultTool, Input Video: {video_path}, Input Question: {question}, "
            f"Output Answer: {result}"
        )
        return result

class InpaintingTool:
    def __init__(self, device, config):
        self.device = device
        self.inpainter = Inpainter(device=device)

    @prompts(
        name = "InpaintingTool",
        description = "Useful when user want to inpaint something in the video."
        "The input to this tool must be a string for the video path, and a string for the question. Concatenate them using # as the separator."
        "For example: the input is ./videos/xxx.mp4#Inpaint the men on the right.",
    )
    def inference(self, input):
        if "#" in input:
            tmp = input.split("#")
            if len(tmp) == 2:
                video_path = tmp[0]
                question = tmp[1]
            else: 
                result = "There is an error in the input of the tool, please check the input and try again."
                return result
        else:
            result = "There is an error in the input of the tool, please check the input and try again."
            return result

        video_name = os.path.basename(video_path).split(".")[0]
        imgseq_path = f"./demo/rvos_res/images/{video_name}/"
        maskseq_path = f"./demo/rvos_res/results/{video_name}/"

        # try:
        res_path = self.inpainter.main_worker(video_path = imgseq_path, mask_path = maskseq_path)
        result = f"Finish inpainting. The results are saved in {res_path}."
        # except:
        #     result = "There is an error. Try to ask the question in a different way."

        print(
            f"\nProcessed InpaintingTool, Input Video: {video_path}, Input Question: {question}, "
            f"Output Answer: {result}"
        )
        return result


############Memory Bulider#########
class VideoTemporalUnderstanding:
    def __init__(self, device, config):
        self.config = config
        self.device = device
        self.basemodel = TemporalBase(device=self.device, config=self.config)

    @prompts(
        name="VideoTemporalUnderstanding",
        description="useful when you need to process temporal information in videos."
        "The input to this tool must be a string for the video path, and a string for the question. Concatenate them using # as the separator."
        "For example: the input is ./videos/xxx.mp4#Are the men happy today?",
    )
    def inference(self, input):
        if "#" in input:
            tmp = input.split("#")
            if len(tmp) == 2:
                video_path = tmp[0]
                question = tmp[1]
            else: 
                result = "There is an error in the input of the tool, please check the input and try again."
                return result
        else:
            result = "There is an error in the input of the tool, please check the input and try again."
            return result
        
        step = self.config.memory.step
        self.basemodel.run_on_video(video_path, step)
        result = "successfully built"
        print(
            f"\nProcessed VideoTemporalUnderstanding, Input Video: {video_path}, "
        )
        return result


class VideoInstanceUnderstanding:
    def __init__(self, device, config):
        self.config = config
        self.device = device
        self.basemodel = InstanceBase(device=self.device, config=self.config)

    @prompts(
        name="VideoInstanceUnderstanding",
        description="useful when you need to understand the instance information in videos."
        "The input to this tool must be a string for the video path, and a string for the question. Concatenate them using # as the separator."
        "For example: the input is ./videos/xxx.mp4#Are the men happy today?",
    )
    def inference(self, input):
        if "#" in input:
            tmp = input.split("#")
            if len(tmp) == 2:
                video_path = tmp[0]
                question = tmp[1]
            else: 
                result = "There is an error in the input of the tool, please check the input and try again."
                return result
        else:
            result = "There is an error in the input of the tool, please check the input and try again."
            return result
        
        step = self.config.memory.step
        self.basemodel.run_on_video(video_path,question,step)
        result = "successfully built"
        print(
            f"\nProcessed VideoInstanceUnderstanding, Input Video: {video_path}, "
        )
        return result



class MemeryBuilder:
    def __init__(self, load_dict, config):
        print(f"Initializing DoraemonGPT, load_dict={load_dict}")

        self.config = config
        self.models = {}
        self.examplesel = CustomExampleSelector()

        for class_name, device in load_dict.items():
            self.models[class_name] = globals()[class_name](device=device,config=self.config)

        # Load Template Foundation Models
        for class_name, module in globals().items():
            if getattr(module, "template_model", False):
                template_required_names = {
                    k
                    for k in inspect.signature(module.__init__).parameters.keys()
                    if k != "self"
                }
                loaded_names = set([type(e).__name__ for e in self.models.values()])
                if template_required_names.issubset(loaded_names):
                    self.models[class_name] = globals()[class_name](
                        **{name: self.models[name] for name in template_required_names}
                    )

        print(f"All the Available Functions: {self.models}")

        self.tools = []
        for instance in self.models.values():
            for e in dir(instance):
                if e.startswith("inference"):
                    func = getattr(instance, e)
                    self.tools.append(
                        Tool(name=func.name, description=func.description, func=func)
                    )

        self.llm = OpenAI(
            openai_api_key = self.config.openai.GPT_API_KEY, 
            model_name = self.config.openai.AGENT_GPT_MODEL_NAME, 
            base_url=self.config.openai.PROXY,
            temperature = 0
        )

        self.memory = ConversationBufferMemory(memory_key="chat_history")

    def init_db_agent(self):
        tools = []
        self.db_model_list = []

        memory_load_dict = {e.split("_")[0].strip(): e.split("_")[1].strip() for e in conf.memory.memory_list}
        for class_name, device in memory_load_dict.items():
            self.db_model_list.append(globals()[class_name](device=device,config=self.config))
        
        for instance in self.db_model_list:
            for e in dir(instance):
                if e.startswith("inference"):
                    func = getattr(instance, e)
                    tools.append(
                        Tool(name=func.name, description=func.description, func=func)
                    )

        llm = OpenAI(
            openai_api_key=self.config.openai.GPT_API_KEY, 
            model_name=self.config.openai.GPT_MODEL_NAME, 
            temperature=0
        )

        memory = ConversationBufferMemory(memory_key="chat_history")
        self.db_agent = initialize_agent(
            tools,
            llm,
            agent="conversational-react-description",
            verbose=True,
            memory=memory,
        )

    def run_db_agent(self, video_path, question,with_two_mem):
        video_dir = os.path.dirname(video_path)
        video_name = os.path.basename(video_path).split(".")[0]
        sql_path = os.path.join(video_dir, video_name + ".db")

        if os.path.exists(sql_path):
            os.remove(sql_path)

        if with_two_mem:
            self.db_model_list[0].inference(video_path + "#"+ question)
            self.db_model_list[1].inference(video_path + "#"+ question)
        else:
            Human_prompt = f"provide a video from {video_path}. You must use at least one tool to finish following tasks, rather than directly imagine from my description. If you understand, say 'Received'."
            AI_prompt = f"Received."
            self.db_agent.memory.save_context(
                {"input": Human_prompt}, {"output": AI_prompt}
            )

            self.db_agent.run(input=question.strip())

            video_dir = os.path.dirname(video_path)
            video_name = os.path.basename(video_path).split(".")[0]
            self.sql_path = os.path.join(video_dir, video_name + ".db")
            conn = sqlite3.connect(self.sql_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='instancedb';"
            )
            rows_1 = cursor.fetchall()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='temporaldb';"
            )
            rows_2 = cursor.fetchall()
            if len(rows_1) == 0 and len(rows_2) == 0:
                self.db_model_list[0].inference(video_path + "#"+ question)
                self.db_model_list[1].inference(video_path + "#"+ question)

    def run_example(self, example):
        input = example["Input"]
        output = example["Output"]
        Human_prompt = f"Here is an example. The question is: {input}; The chain is:{output}; If you understand, say 'Received'."
        AI_prompt = f"Received."

        self.agent.memory.save_context({"input": Human_prompt}, {"output": AI_prompt})

        print(f" Current Memory: {self.agent.memory.load_memory_variables({})}")



def run_a_video(
    MemoryBuilder,
    Planner,
    video_name,
    question,
    possible_anwsers=[],
    skip_mem_build=True,
    with_two_mem = True,
    use_example=False,
    max_answer=1,
    max_try=7,
):
    if (
        not skip_mem_build
    ):  # if you have built the memory, you can skip this step by setting build_mem=False
        MemoryBuilder.init_db_agent()
        MemoryBuilder.run_db_agent(video_name, question,with_two_mem)

    anwsers = Planner.run(
        video_name,
        question,
        possible_anwsers=possible_anwsers,
        max_answer=max_answer,
        max_try=max_try,
        use_example=use_example,
    )
    print("Input video: ", video_name)
    print("Input question: ", question)
    print("The anwsers are:", anwsers)
    print("Total action steps: ", Planner.total_step)
    return anwsers

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="demo")               
    parser.add_argument('--config', default="config/demo_1.yaml",type=str)                           
    opt = parser.parse_args()

    vq_conf = OmegaConf.load(opt.config)
    conf = OmegaConf.load(vq_conf.inference_config_path)

    seed_everything(vq_conf.seed) 

    load_dict = {e.split("_")[0].strip(): e.split("_")[1].strip() for e in conf.tool.tool_list}

    bot = MemeryBuilder(load_dict=load_dict, config=conf)
    planner = ReThinking(
        bot.llm, 
        bot.tools, 
        good_base_reward = conf.mcts_planner.good_base_reward, 
        bad_base_reward = conf.mcts_planner.bad_base_reward, 
        decay_rate = conf.mcts_planner.decay_rate,
    )

    run_a_video(
        bot,
        planner,
        vq_conf.video_path,
        vq_conf.question,
        skip_mem_build = vq_conf.skip_mem_build,
        with_two_mem = vq_conf.with_two_mem,
        max_try = vq_conf.max_try,
        max_answer = vq_conf.max_answer,
    ) 
    
    
    
