# Data Collection Guide

## Objective

Collect webcam data from the real robot camera position so the trained model matches the deployment view as closely as possible.

## Camera Setup Rules

- use the same webcam planned for the robot
- mount it at the real height and tilt angle
- keep the same resolution planned for deployment
- avoid changing field of view after collection starts

## Initial Scene Labels

Use these scene tags in the GUI:

- `stop_wait` when a person is in the path
- `bypass_left` when an object blocks the path and left looks safer
- `bypass_right` when an object blocks the path and right looks safer
- `blocked` when neither side looks safe
- `clear` when the path is open

## Capture Targets

Collect each condition in bright, normal, and dim light when possible:

- empty path
- one person in front
- person crossing left to right
- person crossing right to left
- one object centered
- object on the left side
- object on the right side
- multiple objects
- narrow path
- cluttered background
- near distance
- medium distance
- far distance

## Session Workflow

1. Launch the GUI.
2. Enter a session name such as `warehouse_morning_001`.
3. Enter the real camera name such as `robot_front_cam`.
4. Start the webcam feed.
5. Load a pretrained YOLO model if available.
6. Select the matching scene tag.
7. Write short operator notes when the case is special.
8. Press `Capture Frame` for important frames instead of saving every frame.
9. Open the `Review / Label` tab after collection and correct any wrong boxes before export.

## What To Capture

- frames where the model misses an object
- frames where a person should clearly trigger stop
- frames where an object should clearly trigger bypass
- difficult lighting
- partial occlusion
- scenes with background clutter

## Review Checklist

- remove blurry frames
- remove almost-duplicate frames
- keep difficult failure cases
- keep enough negative samples with no obstacle
- keep balanced counts between `person` and `obstacle`

## Labeling Guidance

For detection labels:

- `person` for humans who must trigger stop/wait
- `obstacle` for static objects that the robot may need to bypass

For now, do not over-split obstacle classes unless the bypass behavior depends on the exact object type.
