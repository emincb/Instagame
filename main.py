import pygetwindow as gw
from pygetwindow import Win32Window
import mouse
from mouse import LEFT
import keyboard as kb
import time
import mss
import numpy as np
import math
import ctypes
import threading
import sys
import json
from collections import deque

que = deque(maxlen=5)  # Reduced queue size for faster response

with open('config.json') as file:
    config = json.load(file)

window_title = config['window_title']
window_width = config['window_width']
window_height = config['window_height']

ball_width = 40
platform_height = 65

left_boundary = 19
right_boundary = window_width - 19
top_boundary = 54
bottom_boundary = window_height - 90

HWND_TOP = 0
SWP_NOSENDCHANGING = 0x0400

def set_window_pos(window_handle, x, y, width, height):
    ret = ctypes.windll.user32.SetWindowPos(
        window_handle,
        HWND_TOP,
        x, y, width, height,
        SWP_NOSENDCHANGING
    )
    return ret

def init():
    all_windows = gw.getAllWindows()
    for w in all_windows:
        if window_title in w.title:
            break
    
    if window_title not in w.title:
        print('No Instagram window found')
        return None

    window = Win32Window(w._hWnd)
    window.activate()

    print(window.size)
    time.sleep(0.2)

    mouse.move(
        x=window.topleft.x + window_width // 2,
        y=window.topleft.y + window_height - 65,
    )
    mouse.press(LEFT)
    time.sleep(0.2)
    mouse.release(LEFT)
    time.sleep(0.2)
    mouse.press(LEFT)

    return window

def rgb_ansi_text(text, r, g, b):
    return f"\033[38;2;{r};{g};{b}m{text}\033[0m"

def get_screenshot(window):
    region = {
        "top": window.topleft.y,
        "left": window.topleft.x,
        "width": window_width,
        "height": window_height
    }

    with mss.mss() as sct:
        while True:
            if kb.is_pressed('space'):
                return
            screenshot = np.array(sct.grab(region))[:, :, :3]
            que.appendleft(screenshot)

def forward_wall(walls, x, y):
    for wall in walls:
        if wall == 'L':
            x = 2 * left_boundary - x
        elif wall == 'R':
            x = 2 * right_boundary - x
        elif wall == 'T':
            y = 2 * top_boundary - y
    return x, y

def reverse_wall(walls, x, y):
    for wall in walls[::-1]:
        if wall == 'L':
            x = 2 * left_boundary - x
        elif wall == 'R':
            x = 2 * right_boundary - x
        elif wall == 'T':
            y = 2 * top_boundary - y
    return x, y

def predict(window):
    prev_dir = 'up'
    prev = [float('inf'), float('inf')]
    sequence = []
    reflected = []
    walls = []
    prev_reflected = []
    prev_time = time.time()
    times = []

    target = np.array([53, 67, 243])
    
    while True:
        if kb.is_pressed('space'):
            return
        
        if len(que) == 0:
            continue

        screenshot = que.pop()

        # Ball detection
        diff = np.abs(screenshot - target)
        mask = np.all(diff < 30, axis=-1)
        
        image_y, image_x = np.where(mask)
        
        if len(image_x) < 30 or len(image_y) < 30:
            continue

        ball_x = int(np.percentile(image_x, 50))
        ball_y = int(np.percentile(image_y, 50))

        if abs(ball_x - prev[0]) < 3 and abs(ball_y - prev[1]) < 3:
            continue

        # Direction detection
        if ball_y > prev[1] or (ball_y == prev[1] and prev_dir == 'up'):
            prev_dir = 'down'
        else:
            if prev_dir == 'down':
                x = reflected[-1][0]
                while x < left_boundary or x > right_boundary:
                    if x < left_boundary:
                        x = 2 * left_boundary - x
                    elif x > right_boundary:
                        x = 2 * right_boundary - x
                y = reflected[-1][1]
                y = 2 * top_boundary - y
                y = 2 * bottom_boundary - y
                
                offset = [x - reflected[-1][0], y - reflected[-1][1]]
                prev_reflected = [[x + offset[0], y + offset[1]] for x, y in reflected[:-1]]
                
                sequence = []
                reflected = []
                walls = []
            prev_dir = 'up'
        
        sequence.append([ball_x, ball_y])

        # Wall detection
        if len(sequence) > 3:
            if sequence[-3][0] > sequence[-2][0] and sequence[-2][0] <= sequence[-1][0]:
                walls.append('L')
            if sequence[-3][0] < sequence[-2][0] and sequence[-2][0] >= sequence[-1][0]:
                walls.append('R')
            
            if len(walls) > 1 and walls[-1] == walls[-2]:
                walls.pop()

            if sequence[-3][1] > sequence[-2][1] and sequence[-2][1] <= sequence[-1][1]:
                if 'T' not in walls:
                    walls.append('T')
            
            if len(walls) > 1 and walls[-1] == walls[-2]:
                walls.pop()
            
            reflected_x, reflected_y = reverse_wall(walls, sequence[-1][0], sequence[-1][1])
            reflected.append([reflected_x, reflected_y])
        else:
            reflected.append([ball_x, ball_y])

        prev = [ball_x, ball_y]

        # Prediction
        if len(reflected) > 2:
            # Use more points for prediction when ball is near bottom for better accuracy
            is_near_bottom = ball_y > (window_height - 200)
            points_to_use = min(len(reflected), 10) if is_near_bottom else 3
            
            # When near bottom, also include previous reflected points for better prediction
            if is_near_bottom and len(prev_reflected) > 0:
                prediction_points = prev_reflected[-5:] + reflected[-points_to_use:]
            else:
                prediction_points = reflected[-points_to_use:]
            
            X = np.array([y for _, y in prediction_points])
            y = np.array([x for x, _ in prediction_points])

            m, b = np.polyfit(X, y, 1)
            
            # Always predict to bottom boundary for consistency
            predicted_x = m * bottom_boundary + b

            while predicted_x < left_boundary or predicted_x > right_boundary:
                if predicted_x < left_boundary:
                    predicted_x = 2 * left_boundary - predicted_x
                elif predicted_x > right_boundary:
                    predicted_x = 2 * right_boundary - predicted_x

            predicted_x = max(left_boundary, min(right_boundary, predicted_x))

            if not math.isnan(predicted_x):
                mouse.move(
                    window.topleft.x + predicted_x,
                    window.topleft.y + window_height - platform_height,
                    absolute=True
                )

        # Performance monitoring
        current_time = time.time()
        times.append(current_time - prev_time)
        prev_time = current_time
        if len(times) > 60:
            times.pop(0)

        # Display debug information
        line = ''
        col = 150
        for j in range(col):
            x = window_width * j // col
            y = ball_y
            if x < ball_x - ball_width//2 or x > ball_x + ball_width//2:
                color = [53, 67, 243]
            else:
                color = target
            line += rgb_ansi_text('█', *color)

        print(line, end=' ')
        print(f"({ball_x:3d}, {ball_y:3d})", end=' ')
        print(f'fps: {round(len(times) / sum(times))}', end=' ')
        print(f"walls: {', '.join(walls)}", end=' ')
        print()
        sys.stdout.flush()

def main(window):
    producer_thread = threading.Thread(target=lambda: get_screenshot(window))
    consumer_thread = threading.Thread(target=lambda: predict(window))

    producer_thread.start()
    consumer_thread.start()

    producer_thread.join()
    consumer_thread.join()

    mouse.release(LEFT)

if __name__ == "__main__":
    window = init()
    if window is not None:
        main(window)