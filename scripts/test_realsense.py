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
        shm, serial_number=serials[0], enable_depth=False, put_downsample=10
    )
    realsense.start()

    out = realsense.ring_buffer._allocate_empty()
    while True:
        try:
            realsense.get(out=out)
            print(out["step_idx"])

        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()


