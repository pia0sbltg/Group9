from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import requests
from flask_cors import CORS
from openai import OpenAI
import threading
import time
import random
import json
import csv

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
CORS(app, resources={r"/*": {"origins": "*"}})

# Global token limit
TOKEN_LIMIT = 2048

class LMStudioAgent:
    def __init__(self, name, api_url, api_key, model, temperature=0.7, starting_prompt=""):
        self.name = name
        self.client = OpenAI(base_url=api_url, api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.starting_prompt = starting_prompt
        self.history = [
            {"role": "system", "content": starting_prompt}
        ]
        self.stop_event = threading.Event()

    def reset_history(self):
        self.history = [
            {"role": "system", "content": self.starting_prompt}
        ]

    def respond(self, message):
        self.history.append({"role": "user", "content": message})
        
        # Calculate the number of tokens to include in the context
        context_tokens = int(TOKEN_LIMIT * 0.5)
        context = self._get_context(context_tokens)
        
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=context,
                temperature=self.temperature,
                stream=True,
            )

            new_message = {"role": "assistant", "content": ""}
            for chunk in completion:
                if self.stop_event.is_set():
                    break
                if chunk.choices[0].delta.content:
                    new_message["content"] += chunk.choices[0].delta.content
                    socketio.emit('new_message', {'role': self.name, 'content': new_message["content"]})

            self.history.append(new_message)
            
            # Save the final message to a CSV file
            try:
                self.save_message_to_csv(new_message["content"])
            except:
                print("Error saving message to CSV")
                pass

            return new_message["content"]
        except Exception as e:
            print(f"Error during response generation: {e}")
            return ""

    def _get_context(self, context_tokens):
        # Create a context with the last `context_tokens` tokens
        context = []
        total_tokens = 0
        for message in reversed(self.history):
            message_tokens = len(message["content"].split())
            if total_tokens + message_tokens > context_tokens:
                break
            context.insert(0, message)
            total_tokens += message_tokens
        return context

    def save_message_to_csv(self, message):
        with open('agent_responses.csv', mode='a', encoding="utf-8", newline='') as file:
            writer = csv.writer(file)
            writer.writerow([self.name, message])

def load_agents_from_config(config_file, set_name):
    with open(config_file, 'r') as f:
        config = json.load(f)
    agents = []
    for agent_config in config[set_name]:
        agent = LMStudioAgent(
            name=agent_config["name"],
            api_url=agent_config["api_url"],
            api_key=agent_config["api_key"],
            model=agent_config["model"],
            temperature=agent_config["temperature"],
            starting_prompt=agent_config["starting_prompt"]
        )
        agents.append(agent)
    return agents

agents = load_agents_from_config('agents_config.json', 'set1')
stop_event = threading.Event()

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('start_conversation')
def handle_start_conversation(data):
    topic = data['topic']
    socketio.emit('new_message', {'role': 'system', 'content': f"Starting conversation on topic: {topic}"})

    for agent in agents:
        agent.reset_history()
        agent.history.append({"role": "user", "content": topic})
        agent.stop_event.clear()

    stop_event.clear()
    threading.Thread(target=run_conversation, args=(agents, topic)).start()

@socketio.on('switch_agent_set')
def handle_switch_agent_set(data):
    set_name = data['set_name']
    global agents
    agents = load_agents_from_config('agents_config.json', set_name)
    print(f"Switched to agent set: {set_name}")

def run_conversation(agents, initial_message, num_turns=15):
    message = initial_message
    last_agent = None
    
    for _ in range(num_turns):
        if stop_event.is_set():
            break
        available_agents = [agent for agent in agents if agent != last_agent]
        agent = random.choice(available_agents)
        response = agent.respond(message)
        if stop_event.is_set():
            break
        socketio.emit('new_message', {'role': agent.name, 'content': response})
        message = response
        last_agent = agent
        time.sleep(1)

@socketio.on('stop_generation')
def handle_stop_generation():
    stop_event.set()
    for agent in agents:
        agent.stop_event.set()
    socketio.emit('new_message', {'role': 'system', 'content': "Conversation stopped by user"})

if __name__ == '__main__':
    socketio.run(app, debug=True)
