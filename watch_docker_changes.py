import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import os

class ChangeHandler(FileSystemEventHandler):
    def __init__(self):
        self.last_trigger_time = time.time()

    def on_modified(self, event):
        # Ignorer les modifications dans les répertoires de logs
        if "logs" in event.src_path or ".git" in event.src_path or "__pycache__" in event.src_path:
            return

        # Attendre au moins 5 secondes entre les déclenchements
        if time.time() - self.last_trigger_time > 5:
            print(f"Fichier modifié: {event.src_path}. Déclenchement du DAG.")
            os.system(
                'curl -X POST http://localhost:8080/api/v1/dags/deploy_code_update_docker/dagRuns '
                '-H "Content-Type: application/json" '
                '-H "Authorization: Basic YWlyZmxvdzplc3NhaUBhaXJmbG93" '
                '-d \'{"conf": {}}\''
            )
            self.last_trigger_time = time.time()

if __name__ == "__main__":
    path = "."  # Répertoire à surveiller 
    event_handler = ChangeHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True) 
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
