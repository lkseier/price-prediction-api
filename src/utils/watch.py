import time
import os
import sys
from watchdog.observers import Observer

# Add the src directory to the Python path
project_root = os.path.abspath(".")

# Import the ReloadHandler class from utils
from utils.reload_handler import ReloadHandler

if __name__ == "__main__":
    # Path to the main script to run
    SCRIPT_TO_RUN = "src/main.py"

    # Directory to watch recursively
    WATCH_PATH = "src"

    # Initialize the observer and event handler
    handler = ReloadHandler(SCRIPT_TO_RUN)
    observer = Observer()
    observer.schedule(handler, WATCH_PATH, recursive=True)
    observer.start()

    print(f"Watching for changes in: {os.path.abspath(WATCH_PATH)}")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        if handler.process:
            handler.process.terminate()
        print("Watcher stopped.")
