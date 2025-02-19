from typing import Literal
from pathlib import Path
import numpy as np
import subprocess as sp
import shlex


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
        # self.loglevel = "-loglevel info"
        self.loglevel = "-loglevel quiet"
        # self.x265_params = '-x265-params "lossless=1 -p=ultrafast -tune=zerolatency"'
        # self.x265_params = '-x265-params "lossless=1 -tune=zerolatency"'
        self.x265_params = "-preset lossless"
        if VideoRecorder.hevc_nvenc_counter <= VideoRecorder.hevc_nvenc_limit:
            VideoRecorder.hevc_nvenc_counter += 1
            encoder = "hevc_nvenc"
        else:
            encoder = "hevc"

        self.get_command = (
            lambda path: f"ffmpeg {self.loglevel} -threads 1 -y -s {self.width}x{self.height}  -pixel_format {input_pix_fmt} -f rawvideo -r {self.fps} -i pipe: -vcodec {encoder} -pix_fmt yuv420p {path} {self.x265_params}"
        )
        self.writer = None
        self.timestamps = []
        self.path = None

    def is_ready(self):
        return self.writer is not None

    def start(self, path: Path | str):
        if isinstance(path, str):
            path = Path(path)
        assert self.writer is None
        self.path = path
        self.writer = sp.Popen(
            shlex.split(self.get_command(path)), stdout=sp.DEVNULL, stdin=sp.PIPE
        )
        self.timestamps = []

    def write_frame(self, data: np.ndarray, frame_time: float):
        assert self.writer is not None
        assert self.width == data.shape[1] and self.height == data.shape[0]
        self.timestamps.append(frame_time)
        self.writer.stdin.write(data.tobytes())  # type: ignore
        self.frame_counter += 1

    def stop(self):
        if self.writer is None:
            return
        self.writer.stdin.close()  # type: ignore
        # self.writer.wait()
        self.writer.terminate()
        self.writer = None
        # with open(self.path.with_suffix(".txt"), "w") as f:
        #     for t in self.timestamps:
        #         f.write(f"{t}\n")
