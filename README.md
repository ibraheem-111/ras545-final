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
ros2 run final object_finder
```


## New developer setup:
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

## How to build
source /opt/ros/humble/setup.bash
colcon build
source install/setup.bash

## For testing Object Finder:
```
ros2 topic pub --once finder_command std_msgs/msg/String "{data: 'Find the red block'}"
```
As an action server
```
ros2 action send_goal /find_object final_interfaces/action/FindObject '{ query: "Where is the packing tape?" }'
```
## For Testing Robot Control 

```
ros2 action send_goal /move_robot final_interfaces/action/MoveRobot '{x: 0.10, y: 0.20, z: 0.30, r: 0.00, motion_type: "joint"}'
```


## For testing orchestrator
```
ros2 topic pub --once /central_query std_msgs/String "{data: 'Pick up the yellow block closest to you and place it on the blue block'}"
```
```
ros2 topic pub --once /central_query std_msgs/String "{data: 'Pick up the yellow block closest to you and place it to the right of the red block to your right'}"
```