import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from main import control_flow
from resume_processing import initialize_storage,INCOMING_DIR

initialize_storage()

WATCH_FOLDER = INCOMING_DIR

class ResumeHandler(FileSystemEventHandler):  #things that should happen when folder event occurs.
    def __init__(self): #timer 
        self.last_triggered = 0
        self.wait_time = 2  # Seconds to wait for more files before starting

    def on_created(self, event): #updates the most_recent updated timestamp.
        if not event.is_directory:
            print(f"File detected: {event.src_path}")
            self.last_triggered = time.time()

    def check_and_run(self):
        # If we have a pending trigger and enough time has passed
        if self.last_triggered > 0 and (time.time() - self.last_triggered) > self.wait_time:
            print("\n--- Folder Quiet. Starting Batch Processing ---")
            control_flow() #main script triggering
            self.last_triggered = 0  # Reset trigger

if __name__ == "__main__":
    event_handler = ResumeHandler()
    observer = Observer()#communicates to the windows to watch the folder
    observer.schedule(event_handler, WATCH_FOLDER, recursive=False)
    observer.start()

    print(f"Monitoring folder: {WATCH_FOLDER}")

    try:
        while True:
            # The loop now checks if it's time to run the stage1 script
            event_handler.check_and_run()
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join() 