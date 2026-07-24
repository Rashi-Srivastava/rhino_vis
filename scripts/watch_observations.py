#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import threading
import time
from pathlib import Path

import h5py
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from scripts.common import configure_logging, load_config, parse_observation_filename


class Processor:
    def __init__(self, config_path: Path):
        self.config_path = config_path.resolve()
        self.config = load_config(self.config_path)
        self.state_path = Path(self.config["paths"]["state_file"])
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.processed = self._load_state()
        self.active = set()
        self.lock = threading.Lock()
        self.execution_lock = threading.Lock()

    def _load_state(self) -> set[str]:
        if not self.state_path.exists():
            return set()
        try:
            return set(json.loads(self.state_path.read_text()).get("processed_files", []))
        except Exception:
            logging.exception("Could not read state file; starting with an empty state.")
            return set()

    def _save_state(self) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"processed_files": sorted(self.processed)}, indent=2))
        tmp.replace(self.state_path)

    def _wait_until_stable(self, path: Path) -> bool:
        checks = int(self.config["processing"]["stable_checks"])
        interval = int(self.config["processing"]["stable_interval_seconds"])
        previous = None
        stable = 0

        while stable < checks:
            if not path.exists():
                return False
            stat = path.stat()
            signature = (stat.st_size, stat.st_mtime_ns)
            if signature == previous and stat.st_size > 0:
                stable += 1
            else:
                stable = 0
                previous = signature
            logging.info("Stability %d/%d for %s", stable, checks, path.name)
            time.sleep(interval)

        try:
            with h5py.File(path, "r") as handle:
                list(handle.keys())
        except Exception:
            logging.warning("File is stable in size but not yet readable as HDF5: %s", path)
            time.sleep(interval)
            return self._wait_until_stable(path)

        return True

    def queue(self, path: Path) -> None:
        path = path.resolve()
        key = str(path)

        try:
            parse_observation_filename(path, self.config["files"]["filename_regex"])
        except ValueError:
            return

        with self.lock:
            if key in self.processed or key in self.active:
                return
            self.active.add(key)

        threading.Thread(target=self._run, args=(path,), daemon=True).start()

    def _run(self, path: Path) -> None:
        key = str(path)
        try:
            if not self._wait_until_stable(path):
                return

            command = [
                sys.executable,
                "-m",
                "scripts.process_observation",
                str(path),
                "--config",
                str(self.config_path),
            ]

            logging.info("Running: %s", " ".join(command))
            with self.execution_lock:
                result = subprocess.run(
                    command,
                    cwd=self.config["paths"]["project_dir"],
                    text=True,
                    capture_output=True,
                    check=False,
                )

            if result.stdout:
                logging.info(result.stdout)
            if result.returncode != 0:
                logging.error("Processing failed for %s\n%s", path.name, result.stderr)
                return

            with self.lock:
                self.processed.add(key)
                self._save_state()
            logging.info("Completed %s", path.name)

        except Exception:
            logging.exception("Unexpected failure for %s", path)
        finally:
            with self.lock:
                self.active.discard(key)


class Handler(FileSystemEventHandler):
    def __init__(self, processor: Processor):
        self.processor = processor

    def on_created(self, event):
        if not event.is_directory:
            self.processor.queue(Path(event.src_path))

    def on_moved(self, event):
        if not event.is_directory:
            self.processor.queue(Path(event.dest_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch for new RHINO observation files.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("/raid1/rhino/rhino_vis/config.yaml"),
    )
    parser.add_argument("--include-existing", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    configure_logging(Path(config["paths"]["logs_dir"]) / "watcher.log")

    input_dir = Path(config["paths"]["input_dir"]).resolve()
    processor = Processor(args.config)
    observer = Observer()
    observer.schedule(Handler(processor), str(input_dir), recursive=False)
    observer.start()

    logging.info("Watching %s", input_dir)

    if args.include_existing:
        for path in sorted(input_dir.glob(config["files"]["pattern"])):
            processor.queue(path)

    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        logging.info("Stopping watcher.")
    finally:
        observer.stop()
        observer.join()


if __name__ == "__main__":
    main()
