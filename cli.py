from src.agentic_patterns.triage_pattern.supervisor_agent import SupervisorAgent
from src.agentic_patterns.planning_pattern.react_agent import ReactAgent
import os
import yaml

def chat_loop():
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), 'src/config/config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # Initialize the triage agent

    web_search_agent = ReactAgent(tools=[], name="web_search_agent", system_prompt="You are a web search agent. You are given a user query and you need to search the web for the most relevant information.")

    agent = SupervisorAgent(config=config, agents=[web_search_agent])
    
    print("Welcome to the AI Bot! Type 'exit' to quit.")
    print("(This bot can help with both general conversations and specific tasks like searching for articles)")
    
    while True:
        # Get user input
        user_input = input("\nYou: ")
        
        if user_input.lower() == 'exit':
            break
        
        try:
            # Process query through triage agent (with verbose mode for demonstration)
            response = agent.run(user_input, verbose=True)
            print(f"\nBot: {response}")
        except Exception as e:
            print(f"\nError: {str(e)}")

def main():
    chat_loop()

if __name__ == "__main__":
    main() 