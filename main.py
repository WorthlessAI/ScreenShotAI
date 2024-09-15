from pynput import keyboard
from pynput.keyboard import Key
import pyautogui

import torch
from torch.nn.functional import softmax

import time
from datetime import datetime

# load model
model = torch.jit.load('screenshot.pth')
model.eval()

CONFIDENCE = 0.75

def main():
    # The event listener will be running in this block
    with keyboard.Events() as events:
        for event in events:
            event = events.get(1.0)
            try:
                print('Alphanumeric key {0} pressed'.format(
                    event.key.char))
            except AttributeError:
                if event is None:
                    continue
                if event.key == keyboard.Key.esc:
                    break
                else:
                    key_idx = [key.name for key in Key].index(str(event.key).split('Key.')[-1])
                    data = torch.zeros(len(Key))
                    data[key_idx] = 1.
                    output = model(data)
                    softmax_output = softmax(output, dim=0)
                    if (softmax_output[1] > CONFIDENCE):
                        screenshot = pyautogui.screenshot()
                        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                        filename = f'screenshot_{timestamp}.png'
                        screenshot.save(filename)
                        print(f'Saved {filename}')

if __name__ == "__main__":
    main()
