from PIL import ImageGrab
import time
import os

def capture_screenshot():
    # Define the folder to store screenshots
    screenshot_folder = os.path.join(os.getcwd(), "screenshots")
    
    # Create folder if it doesn't exist
    if not os.path.exists(screenshot_folder):
        os.makedirs(screenshot_folder)
    
    # Generate a filename with the current timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    screenshot_path = os.path.join(screenshot_folder, f"screenshot_{timestamp}.png")
    
    # Capture the screenshot
    screenshot = ImageGrab.grab()

    # Save the screenshot to a file
    screenshot.save(screenshot_path)

    # Close the screenshot
    screenshot.close()
    
    print(f"Screenshot saved: {screenshot_path}")
