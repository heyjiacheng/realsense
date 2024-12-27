from realsense import SingleRealsense
from multiprocessing.managers import SharedMemoryManager
import time


def main():
    shm = SharedMemoryManager()
    shm.start()
    serials = SingleRealsense.get_connected_devices_serial()
    if not serials:
        print("No connected devices")
        return
    
    realsense = SingleRealsense(
        shm, 
        resolution=(1280, 720),
        serial_number=serials[0], 
        enable_depth=True, 
        # put_downsample=10
    )
    realsense.start()
    realsense.start_recording("scripts/test.mp4")
    time.sleep(5)
    print("Done")
    realsense.stop_recording()
    realsense.stop()
    shm.shutdown()

if __name__ == "__main__":
    main()


