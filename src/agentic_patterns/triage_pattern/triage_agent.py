from openai import OpenAI
from ..tool_pattern.tool_agent import ToolAgent
from ..reflection_pattern.relection_agent import ReflectionAgent
from ..utils.completions import completions_create, ChatHistory, build_prompt_structure
import os


TRIAGE_SYSTEM_PROMPT = """You are a triage agent that determines which specialized agent should handle a user query.
Analyze the user's input and respond in the following XML format:

<decision>
    <agent>TOOL_AGENT or REFLECTION_AGENT</agent>
    <reason>Brief explanation of why this agent was chosen</reason>
</decision>

Guidelines for selection:

<tool_agent_tasks>
- Searching for information or articles
- Performing calculations
- Looking up specific data
- Tasks requiring external tools or APIs
- Queries asking for current information
</tool_agent_tasks>

<reflection_agent_tasks>
- General conversation and discussion
- Analysis and interpretation
- Opinion-based questions
- Complex reasoning
- Theoretical or abstract topics
</reflection_agent_tasks>

Always respond with the XML tags and choose only one agent."""

class TriageAgent:
    def __init__(self, config: dict):
        if 'openai' in self.config and 'api_key' in self.config['openai'] and self.config['openai']['api_key'].startswith('${'):
            env_var = self.config['openai']['api_key'][2:-1]  # Remove ${ and }
            self.config['openai']['api_key'] = os.environ.get(env_var)
            if not self.config['openai']['api_key'] or not self.config['openai']['api_key'].startswith('sk-'):
                raise ValueError(f"Invalid or missing OpenAI API key. Please check your environment variable {env_var}")

        self.client = OpenAI(api_key=config['openai']['api_key'])
        self.model = config['openai']['model']
        self.tool_agent = ToolAgent(config=config)
        self.reflection_agent = ReflectionAgent(config=config)
    
    def _extract_decision(self, response_text: str) -> tuple[str, str]:
        """Extract agent decision and reason from XML response."""
        try:
            # Simple XML parsing (you could use xml.etree.ElementTree for more robust parsing)
            agent = response_text.split('<agent>')[1].split('</agent>')[0].strip()
            reason = response_text.split('<reason>')[1].split('</reason>')[0].strip()
            return agent, reason
        except Exception as e:
            print(f"Error parsing triage response: {e}")
            # Default to reflection agent if parsing fails
            return "REFLECTION_AGENT", "Parsing error, defaulting to reflection agent"

        
    def decide_agent(self, user_input: str) -> tuple[str, str]:
        """Determine which agent should handle the query."""

        triage_history = ChatHistory([
            build_prompt_structure(TRIAGE_SYSTEM_PROMPT, "system"),
            build_prompt_structure(user_input, "user"),
        ])

        response = completions_create(
            client=self.client,
            messages=triage_history,
            model=self.model,
        )
        decision_text = response.choices[0].message.content
        return self._extract_decision(decision_text)

    def run(self, user_input: str, verbose: int = 0) -> tuple[str, str]:
        """
        Run the triage agent.
        
        Args:
            user_input (str): The user's input to be triaged.
            verbose (int, optional): The verbosity level controlling printed output. Defaults to 0.

        Returns:
            tuple[str, str]: The agent and reason for the decision.
        """
        agent, reason = self.decide_agent(user_input)

        if verbose != 0:
            print(f"\nTriage Decision: {agent}")
            print(f"Reason: {reason}\n")
            
        if agent == "TOOL_AGENT":
            response = self.tool_agent.run(user_input)
        else:
            response = self.reflection_agent.run(user_input)
        
        return response