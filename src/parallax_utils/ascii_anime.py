import json
import math
import os
import time


class HexColorPrinter:
    COLOR_MAP = {
        "#000000": ("\033[30m", (0, 0, 0)),
        "#800000": ("\033[31m", (128, 0, 0)),
        "#008000": ("\033[32m", (0, 128, 0)),
        "#808000": ("\033[33m", (128, 128, 0)),
        "#000080": ("\033[34m", (0, 0, 128)),
        "#800080": ("\033[35m", (128, 0, 128)),
        "#008080": ("\033[36m", (0, 128, 128)),
        "#c0c0c0": ("\033[37m", (192, 192, 192)),
        "#808080": ("\033[90m", (128, 128, 128)),
        "#ff0000": ("\033[91m", (255, 0, 0)),
        "#00ff00": ("\033[92m", (0, 255, 0)),
        "#ffff00": ("\033[93m", (255, 255, 0)),
        "#0000ff": ("\033[94m", (0, 0, 255)),
        "#ff00ff": ("\033[95m", (255, 0, 255)),
        "#00ffff": ("\033[96m", (0, 255, 255)),
        "#ffffff": ("\033[97m", (255, 255, 255)),
    }

    RESET = "\033[0m"

    @classmethod
    def hex_to_rgb(cls, hex_color):
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))

    @classmethod
    def color_distance(cls, rgb1, rgb2):
        return math.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(rgb1, rgb2)))

    @classmethod
    def find_closest_color(cls, target_hex):
        target_rgb = cls.hex_to_rgb(target_hex)
        min_distance = float("inf")
        closest_color = "\033[97m"

        for _, (ansi_code, rgb) in cls.COLOR_MAP.items():
            distance = cls.color_distance(target_rgb, rgb)
            if distance < min_distance:
                min_distance = distance
                closest_color = ansi_code

        return closest_color


def clear_screen():
    # Clear screen command for different operating systems
    os.system("cls" if os.name == "nt" else "clear")


def handle_colors_data(raw_data):
    color_dict = {}
    if raw_data is not None:
        config = json.loads(raw_data)
        for key, value in config.items():
            if isinstance(value, str) and value.startswith("#"):
                color_dict[key] = value
    return color_dict


def process_context_color(content, colors):
    res = []
    for row, row_str in enumerate(content):
        processed_row = ""
        for column, text in enumerate(row_str):
            if text in (" ", "#"):
                processed_row += text
                continue
            position_str = str(column) + "," + str(row)
            hex_color = colors.get(position_str, None)
            if hex_color:
                color = HexColorPrinter.find_closest_color(hex_color)
                processed_row += color
            processed_row += text
        processed_row += HexColorPrinter.RESET
        res.append(processed_row)
    return res


def display_ascii_animation(animation_data):
    frames = animation_data.get("frames", [])
    # loop = animation_data.get('loop', False)

    if not frames:
        print("No animation frames found in the JSON data.")
        return

    for frame_data in frames:
        content = frame_data.get("content", None)
        delay = frame_data.get("duration", 30) / 1000.0
        colors_data = frame_data.get("colors", None)
        foreground = colors_data.get("foreground", None)
        colors = handle_colors_data(foreground)

        if content:
            res = process_context_color(content, colors)
            res = "\n".join(res).replace("#", " ")
            clear_screen()
            print(res)
            time.sleep(delay)


def display_parallax_run():
    file_path = "./src/parallax_utils/anime/parallax_run.json"
    try:
        with open(file_path, "r") as f:
            animation_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: The file '{file_path}' contains invalid JSON.")
        return
    display_ascii_animation(animation_data)


def display_parallax_join():
    file_path = "./src/parallax_utils/anime/parallax_run.json"
    try:
        with open(file_path, "r") as f:
            animation_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: The file '{file_path}' contains invalid JSON.")
        return
    display_ascii_animation(animation_data)


if __name__ == "__main__":
    display_parallax_run()
