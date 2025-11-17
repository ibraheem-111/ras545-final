# RAS 545 Robotic Systems 1 Final Project

Agentic Tool calling Dobot Magician with task decomposition

## Environment: 

### To activate the ROS environment and the Python Venv:
``` 
source ./ros_setup.sh
```

### Virtual Env:
``` 
source .venv/bin/activate
```

## Commands:

1. 
```
ros2 run final image_publisher
```

2. 
```
ros2 run final image_subscriber
```

3. Object finder (Gemini)
```
ros2 run

# New developer setup:
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Now they build
source /opt/ros/humble/setup.bash
colcon build
source install/setup.bash
