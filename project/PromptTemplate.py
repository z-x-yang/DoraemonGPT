general_template = {
    "prefix": """Regarding a given video from {video_filename}, answer the following questions as best you can. You have access to the following tools:""",
    "format_instructions": """Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question
""",
    "examples": """Here are some examples:
    {examples}
    """,
    "suffix": """Begin!
Question: {input}{chain_history}
Thought:{thought_prompt}{agent_scratchpad}""",
}
