from typing import Optional, Callable, Dict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import enum
import time
import json
import numpy as np
import pyrealsense2 as rs
import multiprocessing as mp
from threadpoolctl import threadpool_limits
from multiprocessing.managers import SharedMemoryManager
from shared_memory.shared_ndarray import SharedNDArray
from shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from shared_memory.shared_memory_queue import SharedMemoryQueue, Empty
from realsense.ffmpeg_recorder import VideoRecorder
from realsense.timestamp_accumulator import get_accumulate_timestamp_idxs


class Command(enum.Enum):
    SET_COLOR_OPTION = 0
    SET_DEPTH_OPTION = 1
    START_RECORDING = 2
    STOP_RECORDING = 3
    RESTART_PUT = 4


class SingleRealsense(mp.Process):
    MAX_PATH_LENGTH = 4096  # linux path has a limit of 4096 bytes

    def __init__(
        self,
        shm_manager: SharedMemoryManager,
        serial_number,
        resolution=(1280, 720),
        capture_fps=30,
        put_fps=None,
        put_downsample=True,
        record_fps=None,
        enable_color=True,
        enable_depth=False,
        enable_infrared=False,
        get_max_k=60,  # 增加缓冲区大小
        get_time_budget=0.5, # 增加时间预算
        advanced_mode_config=None,
        transform: Optional[Callable[[Dict], Dict]] = None,
        vis_transform: Optional[Callable[[Dict], Dict]] = None,
        recording_transform: Optional[Callable[[Dict], Dict]] = None,
        video_recorder: Optional[VideoRecorder] = None,
        verbose=False,
    ):
        super().__init__()

        if put_fps is None:
            put_fps = capture_fps
        if record_fps is None:
            record_fps = capture_fps

        # create ring buffer
        resolution = tuple(resolution)
        shape = resolution[::-1]
        examples = dict()
        if enable_color:
            examples["color"] = np.empty(shape=shape + (3,), dtype=np.uint8)
        if enable_depth:
            examples["depth"] = np.empty(shape=shape, dtype=np.uint16)
        if enable_infrared:
            examples["infrared"] = np.empty(shape=shape, dtype=np.uint8)
        examples["camera_capture_timestamp"] = 0.0
        examples["camera_receive_timestamp"] = 0.0
        examples["timestamp"] = 0.0
        examples["step_idx"] = 0

        vis_ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=(
                examples if vis_transform is None else vis_transform(dict(examples))
            ),
            get_max_k=1,
            get_time_budget=0.2,
            put_desired_frequency=capture_fps,
        )

        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=examples if transform is None else transform(dict(examples)),
            get_max_k=get_max_k,
            get_time_budget=get_time_budget, # 使用传入的参数
            put_desired_frequency=put_fps,
        )

        # create command queue
        examples = {
            "cmd": Command.SET_COLOR_OPTION.value,
            "option_enum": rs.option.exposure.value,
            "option_value": 0.0,
            "video_path": np.array("a" * self.MAX_PATH_LENGTH),
            "recording_start_time": 0.0,
            "put_start_time": 0.0,
        }

        command_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager, examples=examples, buffer_size=128
        )

        # create shared array for intrinsics
        intrinsics_array = SharedNDArray.create_from_shape(
            mem_mgr=shm_manager, shape=(7,), dtype=np.float64
        )
        intrinsics_array.get()[:] = 0

        if video_recorder is None:
            video_recorder = VideoRecorder(
                width=resolution[0],
                height=resolution[1],
                fps=record_fps,
                input_pix_fmt="bgr24",
            )

        # copied variables
        self.serial_number = serial_number
        self.resolution = resolution
        self.capture_fps = capture_fps
        self.put_fps = put_fps
        self.put_downsample = put_downsample
        self.record_fps = record_fps
        self.enable_color = enable_color
        self.enable_depth = enable_depth
        self.enable_infrared = enable_infrared
        self.advanced_mode_config = advanced_mode_config
        self.transform = transform
        self.vis_transform = vis_transform
        self.recording_transform = recording_transform
        self.video_recorder = video_recorder
        self.verbose = verbose
        self.put_start_time = None
        self.record_start_time = 0.0

        # shared variables
        self.stop_event = mp.Event()
        self.ready_event = mp.Event()
        self.recording_stopped = mp.Event()
        self.ring_buffer = ring_buffer
        self.vis_ring_buffer = vis_ring_buffer
        self.command_queue = command_queue
        self.intrinsics_array = intrinsics_array

    @staticmethod
    def get_connected_devices_serial():
        serials = list()
        for d in rs.context().devices:
            if d.get_info(rs.camera_info.name).lower() != "platform camera":
                serial = d.get_info(rs.camera_info.serial_number)
                product_line = d.get_info(rs.camera_info.product_line)
                # 支持D400和D400系列
                if product_line in ["D400", "D405"]:
                    serials.append(serial)
        serials = sorted(serials)
        return serials

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= user API ===========
    def start(self, wait=True, put_start_time=None):
        self.put_start_time = put_start_time
        super().start()
        if wait:
            self.start_wait()

    def stop(self, wait=True):
        self.stop_event.set()
        if wait:
            self.end_wait()

    def start_wait(self):
        self.ready_event.wait()

    def end_wait(self):
        self.join()

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    def get(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k, out=out)

    def get_vis(self, out=None):
        return self.vis_ring_buffer.get(out=out)

    # ========= user API ===========
    def set_color_option(self, option: rs.option, value: float):
        self.command_queue.put(
            {
                "cmd": Command.SET_COLOR_OPTION.value,
                "option_enum": option.value,
                "option_value": value,
            }
        )

    def set_exposure(self, exposure=None, gain=None):
        """
        exposure: (1, 10000) 100us unit. (0.1 ms, 1/10000s)
        gain: (0, 128)
        """

        if exposure is None and gain is None:
            # auto exposure
            self.set_color_option(rs.option.enable_auto_exposure, 1.0)
        else:
            # manual exposure
            self.set_color_option(rs.option.enable_auto_exposure, 0.0)
            if exposure is not None:
                self.set_color_option(rs.option.exposure, exposure)
            if gain is not None:
                self.set_color_option(rs.option.gain, gain)

    def set_white_balance(self, white_balance=None):
        if white_balance is None:
            self.set_color_option(rs.option.enable_auto_white_balance, 1.0)
        else:
            self.set_color_option(rs.option.enable_auto_white_balance, 0.0)
            self.set_color_option(rs.option.white_balance, white_balance)

    def get_intrinsics(self):
        assert self.ready_event.is_set()
        fx, fy, ppx, ppy = self.intrinsics_array.get()[:4]
        mat = np.eye(3)
        mat[0, 0] = fx
        mat[1, 1] = fy
        mat[0, 2] = ppx
        mat[1, 2] = ppy
        return mat

    def get_depth_scale(self):
        assert self.ready_event.is_set()
        scale = self.intrinsics_array.get()[-1]
        return scale

    def allocate_empty(self, k=None):
        return self.ring_buffer._allocate_empty(k=k)

    def start_recording(self, video_path: str, start_time: float = 0.0):
        assert self.enable_color
        path_len = len(video_path.encode("utf-8"))
        if path_len > self.MAX_PATH_LENGTH:
            raise RuntimeError("video_path too long.")
        self.command_queue.put(
            {
                "cmd": Command.START_RECORDING.value,
                "video_path": video_path,
                "recording_start_time": start_time,
            }
        )

    def save_metadata(self, path: Path):
        data = {
            "serial": self.serial_number,
            "K": self.get_intrinsics().tolist(),
            "resolution": self.resolution,
            "timestamps": self.video_recorder.timestamps.copy(),
        }

        with open(path, "w") as f:
            json.dump(data, f)

    def stop_recording(self):
        self.command_queue.put({"cmd": Command.STOP_RECORDING.value})

    def restart_put(self, start_time):
        self.command_queue.put(
            {"cmd": Command.RESTART_PUT.value, "put_start_time": start_time}
        )

    def wait_for_recording_to_stop(self):
        assert self.ready_event.is_set()
        if self.recording_stopped.is_set():
            self.recording_stopped.wait()

    # ========= interval API ===========
    def run(self):
        # limit threads
        threadpool_limits(1)

        w, h = self.resolution
        fps = self.capture_fps
        
        # 首先获取设备信息
        context = rs.context()
        devices = {d.get_info(rs.camera_info.serial_number): d for d in context.query_devices()}
        device_obj = devices.get(self.serial_number)
        
        if device_obj is None:
            print(f"[SingleRealsense {self.serial_number}] Error: Device not found.")
            return
            
        product_name = device_obj.get_info(rs.camera_info.name)
        is_d405 = "D405" in product_name
        if self.verbose:
            print(f"[SingleRealsense {self.serial_number}] Product name: {product_name}, is D405: {is_d405}")

        # 配置流
        rs_config = rs.config()
        rs_config.enable_device(self.serial_number)
        
        # D405特殊处理
        if is_d405:
            # 对于D405，优先深度/红外
            if self.enable_depth:
                rs_config.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
            if self.enable_infrared:
                rs_config.enable_stream(rs.stream.infrared, w, h, rs.format.y8, fps)
            # D405的彩色流可能不可用，尝试启用
            if self.enable_color:
                try:
                    rs_config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
                except RuntimeError as e:
                    if self.verbose:
                        print(f"[SingleRealsense {self.serial_number}] Could not enable color stream for D405: {e}")
                    self.enable_color = False
        else:
            # 其他D400系列相机
            if self.enable_color:
                rs_config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
            if self.enable_depth:
                rs_config.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
            if self.enable_infrared:
                rs_config.enable_stream(rs.stream.infrared, w, h, rs.format.y8, fps)

        try:
            # 启动管道
            pipeline = rs.pipeline()
            pipeline_profile = pipeline.start(rs_config)
            device = pipeline_profile.get_device()
            
            # 设置帧对齐
            if is_d405 and self.enable_depth:
                align = rs.align(rs.stream.depth)  # D405优先深度对齐
            else:
                align = rs.align(rs.stream.color)  # 其他情况优先彩色对齐

            # 设置全局时间
            if is_d405 and self.enable_depth:
                # D405优先使用深度传感器
                sensor_to_set = device.first_depth_sensor()
            elif self.enable_color:
                # 其他情况使用彩色传感器
                sensor_to_set = device.first_color_sensor()
            else:
                sensor_to_set = None
                
            if sensor_to_set:
                try:
                    sensor_to_set.set_option(rs.option.global_time_enabled, 1)
                except RuntimeError as e:
                    print(f"[SingleRealsense {self.serial_number}] Warning: Could not set global_time_enabled: {e}")
            else:
                print(f"[SingleRealsense {self.serial_number}] Warning: Could not get sensor to set global_time_enabled.")

            # 设置高级模式
            if self.advanced_mode_config is not None:
                try:
                    json_text = json.dumps(self.advanced_mode_config)
                    advanced_mode = rs.rs400_advanced_mode(device)
                    advanced_mode.load_json(json_text)
                except RuntimeError as e:
                    print(f"[SingleRealsense {self.serial_number}] Warning: Could not load advanced mode config: {e}")

            # 获取内参
            stream_for_intrinsics = None
            if is_d405:
                # D405优先使用深度流
                if self.enable_depth:
                    try:
                        stream_for_intrinsics = pipeline_profile.get_stream(rs.stream.depth)
                    except RuntimeError:
                        pass
            else:
                # 其他相机优先彩色流
                if self.enable_color:
                    try:
                        stream_for_intrinsics = pipeline_profile.get_stream(rs.stream.color)
                    except RuntimeError:
                        pass
            
            # 回退方案
            if stream_for_intrinsics is None:
                if self.enable_depth:
                    try:
                        stream_for_intrinsics = pipeline_profile.get_stream(rs.stream.depth)
                    except RuntimeError:
                        pass
                elif self.enable_color:
                    try:
                        stream_for_intrinsics = pipeline_profile.get_stream(rs.stream.color)
                    except RuntimeError:
                        pass
            
            if stream_for_intrinsics:
                intr = stream_for_intrinsics.as_video_stream_profile().get_intrinsics()
                order = ["fx", "fy", "ppx", "ppy", "height", "width"]
                for i, name in enumerate(order):
                    self.intrinsics_array.get()[i] = getattr(intr, name)
            else:
                print(f"[SingleRealsense {self.serial_number}] Warning: Could not get intrinsics from any stream.")
            
            # 获取深度比例
            if self.enable_depth:
                depth_sensor = device.first_depth_sensor()
                if depth_sensor:
                    try:
                        depth_scale = depth_sensor.get_depth_scale()
                        self.intrinsics_array.get()[-1] = depth_scale
                    except RuntimeError as e:
                        print(f"[SingleRealsense {self.serial_number}] Warning: Could not get depth scale: {e}")
                else:
                    print(f"[SingleRealsense {self.serial_number}] Warning: Could not get depth sensor for depth_scale.")

            # 相机预热
            if self.verbose:
                print(f"[SingleRealsense {self.serial_number}] Warming up camera...")
                
            # D405需要更长的预热时间
            warmup_frames = 30 if is_d405 else 10
            warmup_timeout = 8000 if is_d405 else 5000
            
            for _ in range(warmup_frames):
                try:
                    warmup_frameset = pipeline.wait_for_frames(timeout_ms=warmup_timeout)
                    if warmup_frameset:
                        break
                except RuntimeError as e:
                    if "Frame didn't arrive" in str(e):
                        continue
                    else:
                        raise
            
            if self.verbose:
                print(f"[SingleRealsense {self.serial_number}] Main loop started.")

            # 频率调节
            put_idx = None
            put_start_time = self.put_start_time
            if put_start_time is None:
                put_start_time = time.time()

            iter_idx = 0
            t_start = time.time()
            consecutive_timeouts = 0
            max_consecutive_timeouts = 10  # 增加最大超时次数
            
            with ThreadPoolExecutor(max_workers=2) as exec:
                while not self.stop_event.is_set():
                    # 动态超时机制
                    timeout_ms = 5000 + (consecutive_timeouts * 1000)
                    timeout_ms = min(timeout_ms, 15000)
                    
                    try:
                        frameset = pipeline.wait_for_frames(timeout_ms=timeout_ms)
                        consecutive_timeouts = 0
                    except RuntimeError as e:
                        if "Frame didn't arrive" in str(e):
                            consecutive_timeouts += 1
                            if consecutive_timeouts <= max_consecutive_timeouts:
                                if self.verbose or consecutive_timeouts <= 3:
                                    print(f"[SingleRealsense {self.serial_number}] Warning: Frame timeout ({consecutive_timeouts}/{max_consecutive_timeouts}). Extending timeout...")
                                continue
                            else:
                                print(f"[SingleRealsense {self.serial_number}] Error: Too many consecutive timeouts ({consecutive_timeouts}). Camera may be disconnected.")
                                break
                        else:
                            raise
                            
                    if not frameset:
                        print(f"[SingleRealsense {self.serial_number}] Warning: Empty frameset.")
                        continue
                        
                    receive_time = time.time()
                    # 对齐帧
                    frameset = align.process(frameset)

                    # 抓取数据
                    data = dict()
                    data["camera_receive_timestamp"] = receive_time
                    data["camera_capture_timestamp"] = frameset.get_timestamp() / 1000
                    
                    # D405彩色处理
                    if is_d405 and self.enable_color:
                        try:
                            # 首先尝试获取彩色帧
                            color_frame = frameset.get_color_frame()
                            if color_frame:
                                color_frame_data = np.asarray(color_frame.get_data())
                                t = color_frame.get_timestamp() / 1000
                            elif self.enable_infrared:
                                # 回退到红外转彩色
                                infrared_frame = frameset.get_infrared_frame()
                                if infrared_frame:
                                    ir_data = np.asarray(infrared_frame.get_data())
                                    # 将红外单通道转为三通道伪彩色
                                    color_frame_data = np.repeat(ir_data[..., np.newaxis], 3, axis=2)
                                    t = infrared_frame.get_timestamp() / 1000
                                else:
                                    if self.verbose:
                                        print(f"[SingleRealsense {self.serial_number}] Warning: No infrared frame available for D405 color fallback.")
                                    color_frame_data = None
                            else:
                                color_frame_data = None
                            
                            if color_frame_data is not None:
                                data["color"] = color_frame_data
                                data["camera_capture_timestamp"] = t
                        except RuntimeError as e:
                            if self.verbose:
                                print(f"[SingleRealsense {self.serial_number}] Error getting D405 color: {e}")
                    else:
                        # 非D405处理
                        if self.enable_color:
                            try:
                                color_frame = frameset.get_color_frame()
                                if color_frame:
                                    data["color"] = np.asarray(color_frame.get_data())
                            except RuntimeError as e:
                                print(f"[SingleRealsense {self.serial_number}] Error getting color frame: {e}")
                    
                    if self.enable_depth:
                        try:
                            depth_frame = frameset.get_depth_frame()
                            if depth_frame:
                                data["depth"] = np.asarray(depth_frame.get_data())
                        except RuntimeError as e:
                            print(f"[SingleRealsense {self.serial_number}] Error getting depth frame: {e}")
                    
                    if self.enable_infrared and not (is_d405 and self.enable_color):
                        try:
                            infrared_frame = frameset.get_infrared_frame()
                            if infrared_frame:
                                data["infrared"] = np.asarray(infrared_frame.get_data())
                        except RuntimeError as e:
                            print(f"[SingleRealsense {self.serial_number}] Error getting infrared frame: {e}")

                    # 应用变换
                    put_data = data.copy()
                    if self.transform is not None:
                        put_data = self.transform(dict(data))

                    if self.put_downsample:
                        # 频率调节
                        local_idxs, global_idxs, put_idx = (
                            get_accumulate_timestamp_idxs(
                                timestamps=[receive_time],
                                start_time=put_start_time,
                                dt=1 / self.put_fps,
                                next_global_idx=put_idx,
                                allow_negative=True,
                            )
                        )

                        for step_idx in global_idxs:
                            put_data["step_idx"] = step_idx
                            put_data["timestamp"] = receive_time
                            try:
                                self.ring_buffer.put(put_data, wait=False)
                            except TimeoutError as e:
                                if self.verbose:
                                    print(
                                        f"[SingleRealsense {self.serial_number}] Dumping data - {e}"
                                    )
                    else:
                        step_idx = int((receive_time - put_start_time) * self.put_fps)
                        put_data["step_idx"] = step_idx
                        put_data["timestamp"] = receive_time
                        try:
                            self.ring_buffer.put(put_data, wait=False)
                        except TimeoutError as e:
                            if self.verbose:
                                print(
                                    f"[SingleRealsense {self.serial_number}] Dumping data - {e}"
                                )

                    # 准备就绪信号
                    if iter_idx == 2:
                        self.ready_event.set()
                        if self.verbose:
                            print(f"[SingleRealsense {self.serial_number}] Camera ready and stable.")

                    # 可视化数据
                    vis_data = data.copy()
                    if self.vis_transform == self.transform:
                        vis_data = put_data
                    elif self.vis_transform is not None:
                        vis_data = self.vis_transform(dict(data))
                    self.vis_ring_buffer.put(vis_data, wait=False)

                    # 录制帧
                    rec_data = data.copy()
                    if self.recording_transform == self.transform:
                        rec_data = put_data
                    elif self.recording_transform is not None:
                        rec_data = self.recording_transform(dict(data))

                    if self.video_recorder.is_ready() and "color" in rec_data:
                        self.video_recorder.write_frame(
                            rec_data["color"],
                            frame_time=receive_time - self.record_start_time,
                        )
                    elif self.video_recorder.is_ready() and "color" not in rec_data and self.verbose:
                         print(f"[SingleRealsense {self.serial_number}] Warning: No color data to record.")

                    # 性能监控
                    t_end = time.time()
                    duration = t_end - t_start
                    frequency = np.round(1 / duration, 1)
                    t_start = t_end
                    if self.verbose:
                        print(f"[SingleRealsense {self.serial_number}] FPS {frequency}")

                    # 处理命令队列
                    try:
                        commands = self.command_queue.get_all()
                        n_cmd = len(commands["cmd"])
                    except Empty:
                        n_cmd = 0

                    # 执行命令
                    for i in range(n_cmd):
                        command = dict()
                        for key, value in commands.items():  # type: ignore
                            command[key] = value[i]
                        cmd = command["cmd"]
                        if cmd == Command.SET_COLOR_OPTION.value:
                            if not is_d405:  # D405不支持彩色选项
                                sensor = device.first_color_sensor()
                                if sensor:
                                    option = rs.option(command["option_enum"])
                                    value = float(command["option_value"])
                                    try:
                                        sensor.set_option(option, value)
                                    except RuntimeError as e:
                                        if self.verbose:
                                            print(f"[SingleRealsense {self.serial_number}] Error setting color option: {e}")
                        elif cmd == Command.SET_DEPTH_OPTION.value:
                            sensor = device.first_depth_sensor()
                            if sensor:
                                option = rs.option(command["option_enum"])
                                value = float(command["option_value"])
                                try:
                                    sensor.set_option(option, value)
                                except RuntimeError as e:
                                    if self.verbose:
                                        print(f"[SingleRealsense {self.serial_number}] Error setting depth option: {e}")
                        elif cmd == Command.START_RECORDING.value:
                            video_path = str(command["video_path"])
                            start_time = command["recording_start_time"]
                            self.recording_stopped.clear()
                            self.record_start_time = start_time
                            self.video_recorder.start(video_path)
                        elif cmd == Command.STOP_RECORDING.value:
                            self.video_recorder.stop()
                            if self.video_recorder.path is not None:
                                exec.submit(
                                    self.save_metadata,
                                    self.video_recorder.path.with_suffix(".json"),
                                )
                            put_idx = None
                        elif cmd == Command.RESTART_PUT.value:
                            put_idx = None
                            put_start_time = command["put_start_time"]

                    iter_idx += 1
        except Exception as e:
            print(f"[SingleRealsense {self.serial_number}] Exception in run loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.video_recorder.stop()
            try:
                if 'pipeline' in locals() and pipeline:
                    pipeline.stop()
            except Exception as e:
                print(f"[SingleRealsense {self.serial_number}] Error stopping pipeline: {e}")
            self.ready_event.set()

        if self.verbose:
            print(f"[SingleRealsense {self.serial_number}] Exiting worker process.")