from typing import Literal
from pathlib import Path
import numpy as np
import subprocess as sp
import shlex
import time


class VideoRecorder:
    # nvidia cards have a limit of 3 or 5 nvenc sessions.
    # This can be patched to inf https://github.com/keylase/nvidia-patch
    hevc_nvenc_counter: int = 0
    hevc_nvenc_limit: int = 5

    def __init__(
        self,
        width: int,
        height: int,
        input_pix_fmt: Literal["rgb24", "bgr24"],
        fps: int = 30,
    ):
        self.writer = None
        self.fps = fps
        self.width = width
        self.height = height
        self.frame_counter = 0
        self.input_pix_fmt = input_pix_fmt
        self.loglevel = "-loglevel info"  # 改为info级别以便调试
        
        # 使用更简单的编码器设置
        if VideoRecorder.hevc_nvenc_counter < VideoRecorder.hevc_nvenc_limit:
            VideoRecorder.hevc_nvenc_counter += 1
            self.encoder = "h264_nvenc"  # 使用H264而不是HEVC
            self.encoder_params = "-preset p1 -tune ll"  # 低延迟预设
        else:
            self.encoder = "libx264"  # 使用CPU编码
            self.encoder_params = "-preset ultrafast -tune zerolatency"

        self.get_command = (
            lambda path: (
                f"ffmpeg {self.loglevel} "
                f"-f rawvideo -vcodec rawvideo "
                f"-s {self.width}x{self.height} "
                f"-pix_fmt {input_pix_fmt} "
                f"-r {self.fps} "
                f"-i pipe: "
                f"-c:v {self.encoder} {self.encoder_params} "
                f"-pix_fmt yuv420p "
                f"-y {path}"
            )
        )
        self.writer = None
        self.timestamps = []
        self.path = None

    def is_ready(self):
        return self.writer is not None and self.writer.poll() is None

    def start(self, path: Path | str):
        if isinstance(path, str):
            path = Path(path)
        if self.writer is not None:
            self.stop()
            
        self.path = path
        # 确保目录存在
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            command = self.get_command(path)
            print(f"Starting FFmpeg with command: {command}")  # 打印命令以便调试
            
            self.writer = sp.Popen(
                shlex.split(command),
                stdin=sp.PIPE,
                stdout=sp.PIPE,
                stderr=sp.PIPE,
                bufsize=10**8  # 使用更大的缓冲区
            )
            self.timestamps = []
        except sp.SubprocessError as e:
            print(f"Failed to start FFmpeg: {e}")
            self.writer = None
            raise

    def write_frame(self, data: np.ndarray, frame_time: float):
        if not self.is_ready():
            raise RuntimeError('FFmpeg process is not ready or has terminated')
            
        assert self.width == data.shape[1] and self.height == data.shape[0]
        
        try:
            self.timestamps.append(frame_time)
            self.writer.stdin.write(data.tobytes())
            self.writer.stdin.flush()  # 确保数据被写入
            self.frame_counter += 1
        except (BrokenPipeError, IOError) as e:
            print(f"Error writing frame: {e}")
            if self.writer.stderr:
                error = self.writer.stderr.read()
                if error:
                    print(f"FFmpeg error output: {error.decode()}")
            self.stop()
            raise

    def stop(self):
        if self.writer is None:
            return
            
        try:
            if self.writer.stdin:
                self.writer.stdin.flush()
                self.writer.stdin.close()
            
            # 等待进程结束，但设置超时
            try:
                self.writer.wait(timeout=5)
                # 打印FFmpeg的输出
                if self.writer.stderr:
                    error = self.writer.stderr.read()
                    if error:
                        print(f"FFmpeg output: {error.decode()}")
            except sp.TimeoutExpired:
                print("FFmpeg process did not terminate, forcing...")
                self.writer.kill()
                
        except Exception as e:
            print(f"Error during FFmpeg shutdown: {e}")
        finally:
            self.writer = None

    def __del__(self):
        self.stop()
        # decrement the hevc_nvenc_counter
        VideoRecorder.hevc_nvenc_counter -= 1