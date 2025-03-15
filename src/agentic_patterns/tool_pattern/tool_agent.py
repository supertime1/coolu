import json
import re

from colorama import Fore
from dotenv import load_dotenv
from openai import OpenAI
import os
import importlib
from typing import List
from .tool import Tool, validate_arguments
from ..utils.completions import (
    build_prompt_structure,
    ChatHistory,
    completions_create,
    update_chat_history
)
from ..utils.extraction import extract_tag_content

load_dotenv()


TOOL_SYSTEM_PROMPT = """
You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags.
You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug
into functions. Pay special attention to the properties 'types'. You should use those types as in a Python dict.
For each function call return a json object with function name and arguments within <tool_call></tool_call>
XML tags as follows:

<tool_call>
{"name": <function-name>,"arguments": <args-dict>,  "id": <monotonically-increasing-id>}
</tool_call>

Here are the available tools:

<tools>
%s
</tools>
"""


class ToolAgent:
    """
    The ToolAgent class represents an agent that can interact with a language model and use tools
    to assist with user queries. It generates function calls based on user input, validates arguments,
    and runs the respective tools.

    Attributes:
        tools (Tool | list[Tool]): A list of tools available to the agent.
        model (str): The model to be used for generating tool calls and responses.
        client (OpenAI): The OpenAI client used to interact with the language model.
        tools_dict (dict): A dictionary mapping tool names to their corresponding Tool objects.
    """

    def __init__(self, config: dict) -> None:
        self.client = OpenAI()
        self.config = config
        # Replace environment variables in config - update path to nested openai.api_key
        if 'openai' in self.config and 'api_key' in self.config['openai'] and self.config['openai']['api_key'].startswith('${'):
            env_var = self.config['openai']['api_key'][2:-1]  # Remove ${ and }
            self.config['openai']['api_key'] = os.environ.get(env_var)
            if not self.config['openai']['api_key'] or not self.config['openai']['api_key'].startswith('sk-'):
                raise ValueError(f"Invalid or missing OpenAI API key. Please check your environment variable {env_var}")

        self.client = OpenAI(api_key=config['openai']['api_key'])
        self.model = config['openai']['model']

        # load tools
        self.tools = self.load_tools()
        self.tools_dict = {tool.name: tool for tool in self.tools}

    def load_tools(self) -> list[Tool]:
        """
        Loads the tools from the tools directory.
        
        Returns:
            List[Tool]: List of available tools
        """
        tools = []
        tools_dir = os.path.join(os.path.dirname(__file__), 'tools')
        
        # Skip __init__.py and __pycache__
        for file in os.listdir(tools_dir):
            if file.endswith('.py') and not file.startswith('__'):
                # Convert filename to module path
                module_name = f"src.agentic_patterns.tool_pattern.tools.{file[:-3]}"
                try:
                    # Import the module
                    module = importlib.import_module(module_name)
                    # Get the tools list from the module
                    if hasattr(module, 'tools'):
                        tools.extend(module.tools)
                except Exception as e:
                    print(f"Error loading tools from {file}: {e}")
        
        return tools
    
    def add_tool_signatures(self) -> str:
        """
        Collects the function signatures of all available tools.

        Returns:
            str: A concatenated string of all tool function signatures in JSON format.
        """
        return "".join([tool.fn_signature for tool in self.tools])

    def process_tool_calls(self, tool_calls_content: list) -> dict:
        """
        Processes each tool call, validates arguments, executes the tools, and collects results.

        Args:
            tool_calls_content (list): List of strings, each representing a tool call in JSON format.

        Returns:
            dict: A dictionary where the keys are tool call IDs and values are the results from the tools.
        """
        observations = {}
        for tool_call_str in tool_calls_content:
            tool_call = json.loads(tool_call_str)
            tool_name = tool_call["name"]
            tool = self.tools_dict[tool_name]

            print(Fore.GREEN + f"\nUsing Tool: {tool_name}")

            # Validate and execute the tool call
            validated_tool_call = validate_arguments(
                tool_call, json.loads(tool.fn_signature)
            )
            print(Fore.GREEN + f"\nTool call dict: \n{validated_tool_call}")

            result = tool.run(**validated_tool_call["arguments"])
            print(Fore.GREEN + f"\nTool result: \n{result}")

            # Store the result using the tool call ID
            observations[validated_tool_call["id"]] = result

        return observations

    def run(self, user_msg: str) -> str:
        """
        Handles the full process of interacting with the language model and executing a tool based on user input.

        Args:
            user_msg (str): The user's message that prompts the tool agent to act.

        Returns:
            str: The final output after executing the tool and generating a response from the model.
        """
        user_prompt = build_prompt_structure(prompt=user_msg, role="user")

        tool_chat_history = ChatHistory(
            [
                build_prompt_structure(
                    prompt=TOOL_SYSTEM_PROMPT % self.add_tool_signatures(),
                    role="system",
                ),
                user_prompt,
            ]
        )
        agent_chat_history = ChatHistory([user_prompt])

        tool_call_response = completions_create(
            self.client, messages=tool_chat_history, model=self.model
        )
        tool_calls = extract_tag_content(str(tool_call_response), "tool_call")

        if tool_calls.found:
            observations = self.process_tool_calls(tool_calls.content)
            update_chat_history(
                agent_chat_history, f'f"Observation: {observations}"', "user"
            )

        return completions_create(self.client, agent_chat_history, self.model)