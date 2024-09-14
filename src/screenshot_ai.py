import keyboard  # Library to detect key presses
import time
from screenshot_capture import capture_screenshot

def listen_for_printscreen():
    print("Listening for PrintScreen key press...")
    while True:
        # Check if PrintScreen button is pressed
        if keyboard.is_pressed('c'):
            print("PrintScreen button detected!")
            capture_screenshot()  # Call the function to capture a screenshot
            time.sleep(1)  # Prevent multiple captures from a single press
        time.sleep(0.1)  # Small delay to reduce CPU usage


def main():
    try:
        listen_for_printscreen()  # Start listening for the PrintScreen button
    except KeyboardInterrupt:
        print("Exiting the application...")

if __name__ == "__main__":
    main()