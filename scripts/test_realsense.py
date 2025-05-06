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
    
    print(f"Found devices: {serials}")
    
    realsense = SingleRealsense(
        shm, serial_number=serials[0], 
        enable_depth=False,
        put_downsample=False,  # 禁用降采样
        verbose=True  # 启用详细输出
    )
    realsense.start()

    out = realsense.ring_buffer._allocate_empty()
    while True:
        try:
            realsense.get(out=out)
            print(f"Step Index: {out['step_idx']}, Timestamp: {out['timestamp']:.3f}, Capture Time: {out['camera_capture_timestamp']:.3f}")
            time.sleep(0.1)  # 添加小的延时以便于观察输出

        except KeyboardInterrupt:
            break

    realsense.stop()
    shm.shutdown()


if __name__ == "__main__":
    main()


