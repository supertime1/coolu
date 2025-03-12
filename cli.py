from src.agentic_patterns.reflection_pattern.relection_agent import ReflectionAgent
import asyncio
import os
import yaml

async def chat_loop():
    # Initialize the reflection agent

    config_path = os.path.join(os.path.dirname(__file__), 'src/config/config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    agent = ReflectionAgent(config=config)
    
    print("Welcome to the AI Bot! Type 'exit' to quit.")
    
    while True:
        # Get user input
        user_input = input("\nYou: ")
        
        if user_input.lower() == 'exit':
            break
        
        # Get agent's response with reflection
        response = agent.run(user_input)
        print(f"\nBot: {response}")
        
        # Add agent's response to history
        # reflection_history.append({"role": "assistant", "content": response})

def main():
    asyncio.run(chat_loop())

if __name__ == "__main__":
    main() 