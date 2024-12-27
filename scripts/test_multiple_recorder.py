from realsense import MultiRealsense
import time
from multiprocessing.managers import SharedMemoryManager


def main():
    shm = SharedMemoryManager()
    shm.start()
    realsenses = MultiRealsense(
        shm_manager=shm,
        resolution=(1280, 720),
        enable_depth=False, 
        verbose=False)
    realsenses.start()
    # data = realsenses.allocate_empty()
    realsenses.start_recording(["scripts/test1.mp4", "scripts/test2.mp4", "scripts/test3.mp4"], 0.0)
    time.sleep(4)
    print("Done")
    realsenses.stop_recording()
    realsenses.stop()
    shm.shutdown()


if __name__ == "__main__":
    main()


