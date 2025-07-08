import sys
import subprocess
from watchdog.events import FileSystemEventHandler

class ReloadHandler(FileSystemEventHandler):
    """
    ReloadHandler monitors Python files and restarts a target script when changes are detected.
    """

    def __init__(self, script_path):
        """
        Initialize the handler with the script to be restarted.
        """
        self.script_path = script_path
        self.process = None
        self.restart_script()

    def restart_script(self):
        """
        Start or restart the Python script process.
        """
        if self.process:
            self.process.terminate()
            print("Script restarted.")
        self.process = subprocess.Popen([sys.executable, self.script_path])

    def on_any_event(self, event):
        """
        Called on any filesystem event. If a .py file is changed, restart the script.
        """
        if event.src_path.endswith(".py") and not event.is_directory:
            print(f"Change detected: {event.src_path}")
            self.restart_script()
