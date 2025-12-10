# TD3 for Door Opening Task
This repository contains an implementation of the **Twin Delayed Deep Deterministic Policy Gradient (TD3)** algorithm trained to solve the **Door** opening task in **Robosuite(MuJoCo)** using a Panda Robot Arm.
The goal of the project is for the robot to learn how to **grasp the handle and pull the door oprn**.

## Features
1. Custom TD3 implementation
2. Replay buffer & Exploration noise
3. Training & Evaluation Scripts
4. Tensorboard Logging ('Logs')
5. Robosuite and Gymnasium Wrapper

## Project Structure
td3-robosuite-door/

│

├── main.py # Training script

├── test.py # Evaluation / rendering script

├── td3_torch.py # TD3 Agent implementation

├── networks.py # Actor + Critic networks

├── buffer.py # Replay buffer

│

├── tmp/td3/ # Saved models (auto-created)

├── Logs/ # TensorBoard logs

│

├── .gitignore

└── README.md

## Installation

### 1. Clone the repository 
```bash
git clone https://github.com/ShrreyaRajneesh/td3-robosuite-door.git
cd td3-robosuite-door 
```

### 2. Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate #Linux
# for windows
# .venv\Scripts\activate 
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Training the agent
```bash
python main.py
```
Training logs are automatically saved and can be viewed using TensorBoard
```bash
tensorboard --logdir Logs
```

### 5. Running the trained Policy
```bash
python test.py
```
### 6. Saving & Loading Models
```bash
agent.save_models()
```
or load
```bash
agent.load_models()
```






