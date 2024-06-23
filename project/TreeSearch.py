import pdb
import os
from langchain.prompts import PromptTemplate

import math
import numpy as np

# langchain.debug = True
from langchain.agents.initialize import initialize_agent
from langchain.agents.tools import Tool
from langchain.llms.openai import OpenAI
from langchain.llms.openai import OpenAI
from langchain import OpenAI

from project.ExampleSelector import CustomExampleSelector

from project.PromptTemplate import general_template


def softmax(x):
    if isinstance(x, list):
        x = np.array(x)
    x -= np.max(x)
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x


class SafeDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"


class Node(object):
    def __init__(self, value, father=None):
        self.father = father
        self.value = value
        self.children = []

    def add_child(self, node):
        self.children.append(node)

    def is_leaf(self):
        return len(self.children) == 0

    def get_ancestors(self, remove_root=True, child_first=False):
        ancestors = []
        current = self
        while current.father is not None:
            ancestors.append(current.father)
            current = current.father
        if len(ancestors) == 0:
            return []
        if remove_root:
            ancestors.pop()
        if not child_first:
            ancestors.reverse()
        return ancestors

    def get_descendants(self, remove_leaf=True):
        descendants = []
        if len(self.children) == 0:
            return []
        for child in self.children:
            if remove_leaf and child.is_leaf():
                continue
            descendants.append(child)
            descendants.extend(child.get_descendants(remove_leaf=remove_leaf))
        return descendants


class MCSearchTree(object):
    def __init__(self, value):
        self.root = Node(value)
        self.current = self.root

    def traverse(self):
        self._traverse_helper(self.root)

    def _traverse_helper(self, node):
        print(node.value)
        for child in node.children:
            self._traverse_helper(child)

    def add_child(self, value):
        new_node = Node(value, father=self.current)
        self.current.add_child(new_node)
        self.current = new_node

    def get_ancestors(self, remove_root=True, child_first=False):
        ancestors = self.current.get_ancestors(
            remove_root=remove_root, child_first=child_first
        )
        return ancestors

    def set_current(self, node):
        self.current = node

    def is_root(self):
        return self.current == self.root


class ReThinking(object):
    def __init__(
        self,
        llm,
        tools,
        good_base_reward=1.0,
        bad_base_reward=-1.0,
        decay_rate=0.5,
        template=general_template,
        use_example=False,
    ):
        self.llm = llm
        self.tools = tools
        self.template = template
        self.use_example = use_example
        self.video_name = ""
        self.question = ""
        self.tree = MCSearchTree({"action": None, "observation": "", "reward": 0.0})
        self.examplesel = CustomExampleSelector() if self.use_example else None
        self.examples = ""
        self.total_step = 0
        self.parsing_error_alarm = "Failed to parse"
        self.good_base_reward = good_base_reward
        self.bad_base_reward = bad_base_reward
        self.decay_rate = decay_rate
        self.possible_anwsers = []

    def init_new_tree(self, video_name, question, possible_anwsers=[]):
        self.total_step = 0
        self.possible_anwsers = possible_anwsers
        self.video_name = video_name
        self.question = question
        self.tree = MCSearchTree({"action": None, "observation": "", "reward": 0.0})
        if self.use_example:
            self.examples = self.examplesel.select_examples(question)

    def run(
        self,
        video_name,
        question,
        possible_anwsers=[],
        max_answer=3,
        max_try=8,
        use_example=False,
    ):
        self.use_example = use_example
        self.init_new_tree(video_name, question, possible_anwsers=possible_anwsers)
        anwsers = {"good_anwsers": [], "bad_anwsers": []}
        num_try = 0
        while (
            len(anwsers["good_anwsers"]) + len(anwsers["bad_anwsers"]) < max_answer
            and num_try < max_try
        ):
            num_try += 1
            num_anwser = len(anwsers["good_anwsers"]) + len(anwsers["bad_anwsers"]) + 1
            print(f"\n\nAnswer {num_anwser} - Try {num_try}:\n\n")
            observation, is_good_result = self.expansion()  # 1-step expansion
            if is_good_result:
                answer, is_good_result = self.simulation()
            else:
                answer = observation
            self.backpropagation(is_good_result)
            self.selection()
            if is_good_result:
                anwsers["good_anwsers"].append(answer)
            else:
                anwsers["bad_anwsers"].append(answer)

        return anwsers

    def get_new_agent(self, chain_history="", thought_prompt=""):
        template = self.template
        prefix, instruction, suffix = (
            template["prefix"].format(video_filename=self.video_name),
            template["format_instructions"],
            template["suffix"].format_map(
                SafeDict(
                    chain_history="\n" + chain_history if chain_history else "",
                    thought_prompt=thought_prompt,
                )
            ),
        )

        if self.use_example:
            suffix = (
                template["examples"].format_map(SafeDict(examples=self.examples))
                + suffix
            )

        agent = initialize_agent(
            self.tools,
            self.llm,
            agent="zero-shot-react-description",
            verbose=True,
            handle_parsing_errors=self.parsing_error_alarm,
            agent_kwargs={
                "prefix": prefix,
                "format_instructions": instruction,
                "suffix": suffix,
            },
        )

        return agent

    def get_ancestor_history(self):
        tree_history = ""
        history_nodes = self.tree.get_ancestors(remove_root=True, child_first=False)
        history_nodes.append(self.tree.current)
        if not self.tree.is_root():
            for node in history_nodes:
                tree_history += "\n" + self.format_node_info(node)
        return tree_history

    def get_child_history(self):
        template = """ I have thought about the next action before, such as {tree_history}. I want to think out a different action. Regarding the now state and previous action candidates, I"""
        tree_history = ""
        history_nodes = self.tree.current.children
        if len(history_nodes) > 0:
            for node in history_nodes:
                if tree_history != "":
                    tree_history += ", "
                tree_history += f'"{node.value["action"].tool}" with Input "{node.value["action"].tool_input}" and Observation "{node.value["observation"]}"'
                # tree_history += f'"{node.value["action"].tool}" with input "{node.value["action"].tool_input}"'
            return template.format(tree_history=tree_history)
        else:
            return ""

    def format_node_info(self, node):
        action, observation = node.value["action"], node.value["observation"]
        return f"Thought: {action.log}\nObservation: {observation}"

    def selection(self, sample_all_expandable_nodes=True):
        if sample_all_expandable_nodes:
            all_descendant = self.tree.root.get_descendants(remove_leaf=True)
            nodes = all_descendant + [self.tree.root]
        else:
            ancestors = self.tree.get_ancestors(remove_root=False, child_first=False)
            nodes = ancestors
        rewards = [node.value["reward"] for node in nodes]
        prob = softmax(rewards)
        node_sample = np.random.choice(nodes, p=prob)
        self.tree.set_current(node_sample)

    def expansion(self, max_step=1):
        ancestor_history = self.get_ancestor_history()
        if ancestor_history != "":
            print("ancestor_history:", ancestor_history)
        child_history = self.get_child_history()
        if child_history != "":
            print("child_history:", child_history)
        agent = self.get_new_agent(
            chain_history=ancestor_history, thought_prompt=child_history
        )
        agent_iterator = agent.iter(self.question)

        observation = ""
        is_good_result = True
        for step_idx, step in enumerate(agent_iterator):
            if output := step.get("intermediate_step"):
                self.total_step += 1
                action, observation = output[0]
                self.tree.add_child(
                    {"action": action, "observation": observation, "reward": 0.0}
                )
                if not self.is_good_observation(observation):
                    is_good_result = False
                    break
            if step_idx + 1 >= max_step:
                break
        return observation, is_good_result

    def simulation(self):
        ancestor_history = self.get_ancestor_history()
        agent = self.get_new_agent(chain_history=ancestor_history)
        agent_iterator = agent.iter(self.question)

        is_good_result = True
        for step in agent_iterator:
            if output := step.get("intermediate_step"):
                self.total_step += 1
                action, observation = output[0]
                self.tree.add_child(
                    {"action": action, "observation": observation, "reward": 0.0}
                )
                if not self.is_good_observation(observation):
                    is_good_result = False
                    break
        if not is_good_result:
            return observation, is_good_result

        final_answer = agent_iterator.final_outputs["output"]
        self.tree.current.value["final_anwser"] = final_answer

        if not self.is_good_final_result(final_answer):
            is_good_result = False

        return final_answer, is_good_result

    def is_good_observation(self, observation):
        if self.parsing_error_alarm in observation:
            return False
        return True

    def is_good_final_result(self, final_result):
        if self.parsing_error_alarm in final_result:
            return False
        if len(self.possible_anwsers) > 0:
            for possible_anwser in self.possible_anwsers:
                if possible_anwser in final_result:
                    return True
            return False
        return True

    def backpropagation(self, is_good_result=True):
        ancestors = self.tree.get_ancestors(child_first=True)
        base_reward = self.good_base_reward if is_good_result else -self.bad_base_reward
        step_distance = 0.0
        for node in ancestors:
            step_distance += 1.0
            node.value["reward"] += base_reward * math.exp(
                -self.decay_rate * (step_distance - 1.0)
            )


# if __name__ == "__main__":

#     llm = OpenAI(
#         openai_api_key=GPT_API_KEY, model_name="text-davinci-003", temperature=0
#     )

#     from KnowledgeBase import GoogleSearching

#     google_searching = GoogleSearching()
#     tools = [
#         Tool(
#             name=google_searching.inference.name,
#             description=google_searching.inference.description,
#             func=google_searching.inference,
#         )
#     ]

#     video_filename = f"./demo/nextqa_demo/ch_1.mp4"
#     txt = "How is the baby held by?"
#     possible_anwsers = [
#         "A. The baby is held by the mother.",
#         "B. The baby is held by the father.",
#     ]

#     planner = ReThinking(llm, tools)
#     anwsers = planner.run(
#         video_filename, txt, possible_anwsers=possible_anwsers, max_answer=3, max_try=8
#     )
#     print("The anwsers are:", anwsers)
