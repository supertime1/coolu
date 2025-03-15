from src.agentic_patterns.triage_pattern.triage_agent import TriageAgent
import os
import yaml

def chat_loop():
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), 'src/config/config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # Initialize the triage agent
    agent = TriageAgent(config=config)
    
    print("Welcome to the AI Bot! Type 'exit' to quit.")
    print("(This bot can help with both general conversations and specific tasks like searching for articles)")
    
    while True:
        # Get user input
        user_input = input("\nYou: ")
        
        if user_input.lower() == 'exit':
            break
        
        try:
            # Process query through triage agent (with verbose mode for demonstration)
            response = agent.process_query(user_input, verbose=True)
            print(f"\nBot: {response}")
        except Exception as e:
            print(f"\nError: {str(e)}")

def main():
    chat_loop()

if __name__ == "__main__":
    main() 