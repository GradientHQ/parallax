import json
import time
import os

def clear_screen():
    # Clear screen command for different operating systems
    os.system('cls' if os.name == 'nt' else 'clear')

def display_ascii_animation(animation_data):
    frames = animation_data.get('frames', [])
    # loop = animation_data.get('loop', False)

    if not frames:
        print("No animation frames found in the JSON data.")
        return

    for frame_data in frames:
        frame = frame_data.get('frame')
        delay = frame_data.get('delay', 200) / 1000.0  # Default 200ms delay

        if frame:
            clear_screen()
            print(frame)
            time.sleep(delay)

def display_parallax_run():
    file_path = "./src/parallax_utils/anime/parallax_run.json"
    try:
        with open(file_path, 'r') as f:
            animation_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: The file '{file_path}' contains invalid JSON.")
        return
    display_ascii_animation(animation_data)
