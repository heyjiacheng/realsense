from realsense import SingleRealsense
from multiprocessing.managers import SharedMemoryManager
import time
from pathlib import Path
import os


def main():
    shm = SharedMemoryManager()
    shm.start()
    serials = SingleRealsense.get_connected_devices_serial()
    if not serials:
        print("No connected devices")
        return
    
    print(f"Found devices: {serials}")
    
    output_path = Path("scripts/test.mp4")
    # 如果文件已存在，先删除
    if output_path.exists():
        print(f"Removing existing file: {output_path}")
        output_path.unlink()
    
    realsense = SingleRealsense(
        shm, 
        resolution=(1280, 720),
        serial_number=serials[0], 
        enable_depth=True,
        verbose=True  # 启用详细输出
    )
    
    try:
        print("Starting camera...")
        realsense.start()
        print("Camera started, waiting for ready state...")
        realsense.start_wait()
        
        if not realsense.is_ready:
            print("Camera failed to initialize")
            return
            
        print("Starting recording...")
        realsense.start_recording(str(output_path))
        
        print("Recording for 5 seconds...")
        time.sleep(5)
        print("Stopping recording...")
        realsense.stop_recording()
        
        print("Waiting for recording to complete...")
        realsense.wait_for_recording_to_stop()
        
        # 验证文件是否创建成功
        if output_path.exists():
            size = output_path.stat().st_size
            print(f"Recording completed. File size: {size/1024/1024:.2f} MB")
        else:
            print("Error: Recording file was not created!")
        
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        print("Cleaning up...")
        realsense.stop()
        shm.shutdown()
        print("Cleanup completed")


if __name__ == "__main__":
    main()


