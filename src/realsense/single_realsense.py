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
                if product_line == "D400":
                    # only works with D400 series
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
        
        # First check if the device is available and get its info
        rs_config = rs.config()
        rs_config.enable_device(self.serial_number)
        
        # Get product name to handle D405 specifics
        context = rs.context()
        devices = {d.get_info(rs.camera_info.serial_number): d for d in context.query_devices()}
        device_obj = devices.get(self.serial_number)
        
        if device_obj is None:
            print(f"[SingleRealsense {self.serial_number}] Error: Device not found.")
            return
            
        product_name = device_obj.get_info(rs.camera_info.name)
        print(f"[SingleRealsense {self.serial_number}] Product name: {product_name}")
        
        # Store original requests from constructor for clarity in logic
        original_enable_color_req = self.enable_color
        original_enable_depth_req = self.enable_depth
        original_enable_infrared_req = self.enable_infrared

        if product_name == "Intel RealSense D405":
            # print(f"[SingleRealsense {self.serial_number}] D405: Applying D405-specific stream configuration.")
            # We will modify self.enable_color and self.enable_infrared instance flags based on what we can realistically provide.
            
            d405_provides_depth = False
            d405_provides_color_via_ir = False
            d405_provides_direct_ir = False

            if original_enable_depth_req:
                # print(f"[SingleRealsense {self.serial_number}] D405: Configuring Depth stream ({w}x{h}@{fps}fps).")
                rs_config.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
                d405_provides_depth = True
            
            if original_enable_color_req:
                # print(f"[SingleRealsense {self.serial_number}] D405: Color requested. Will use Infrared stream as color source for stability.")
                # print(f"[SingleRealsense {self.serial_number}] D405: Configuring Infrared stream ({w}x{h}@{fps}fps) for color data.")
                rs_config.enable_stream(rs.stream.infrared, w, h, rs.format.y8, fps)
                self.enable_infrared = True # Ensure IR frame is available for fallback in frame loop
                d405_provides_color_via_ir = True
                d405_provides_direct_ir = True # IR stream is now active
            elif original_enable_infrared_req: # Color not requested, but IR was
                # print(f"[SingleRealsense {self.serial_number}] D405: Infrared stream requested (no color request).")
                # print(f"[SingleRealsense {self.serial_number}] D405: Configuring Infrared stream ({w}x{h}@{fps}fps).")
                rs_config.enable_stream(rs.stream.infrared, w, h, rs.format.y8, fps)
                self.enable_infrared = True # Ensure IR frame is available
                d405_provides_direct_ir = True

            # Update instance's self.enable_color flag based on whether we plan to provide color (via IR)
            self.enable_color = d405_provides_color_via_ir
            # self.enable_depth remains original_enable_depth_req
            # self.enable_infrared has been updated if IR stream was configured

            if not d405_provides_depth and not d405_provides_color_via_ir and not d405_provides_direct_ir:
                print(f"[SingleRealsense {self.serial_number}] D405: WARNING - No streams configured based on requests.")

        else: # For other cameras like D435
            # print(f"[SingleRealsense {self.serial_number}] {product_name}: Applying standard stream configuration.")
            if self.enable_color:
                rs_config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
            if self.enable_depth:
                rs_config.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
            if self.enable_infrared:
                rs_config.enable_stream(rs.stream.infrared, w, h, rs.format.y8, fps)

        try:
            # start pipeline
            pipeline = rs.pipeline()
            pipeline_profile = pipeline.start(rs_config)

            # Get device info from pipeline_profile for robustness
            device = pipeline_profile.get_device()
            
            # Set up frame alignment - MUST be done AFTER pipeline.start() and with correct profile
            # And align target stream must be one that was successfully enabled.
            align_to_stream = None
            if product_name == "Intel RealSense D405":
                if original_enable_depth_req: # Check original request for depth
                    # print(f"[SingleRealsense {self.serial_number}] D405: Aligning to Depth stream.")
                    align_to_stream = rs.stream.depth
                elif original_enable_color_req or original_enable_infrared_req: # If depth not primary, but IR (for color or direct) is
                    # print(f"[SingleRealsense {self.serial_number}] D405: Aligning to Infrared stream.")
                    align_to_stream = rs.stream.infrared
                else:
                    # print(f"[SingleRealsense {self.serial_number}] D405: Warning - No primary alignment stream, defaulting to Depth for align object.")
                    align_to_stream = rs.stream.depth 
            else: # For other cameras, align to color if available, else depth
                if original_enable_color_req:
                    # print(f"[SingleRealsense {self.serial_number}] {product_name}: Aligning to Color stream.")
                    align_to_stream = rs.stream.color
                elif original_enable_depth_req:
                    # print(f"[SingleRealsense {self.serial_number}] {product_name}: Color not enabled, aligning to Depth stream.")
                    align_to_stream = rs.stream.depth
                elif original_enable_infrared_req: # Added this condition for non-D405 IR alignment
                    # print(f"[SingleRealsense {self.serial_number}] {product_name}: Warning - No color/depth for alignment, aligning to Infrared.")
                    align_to_stream = rs.stream.infrared
                else:
                    # print(f"[SingleRealsense {self.serial_number}] {product_name}: Warning - No color/depth/IR for alignment, defaulting to first available or None.")
                    # Attempt to find any enabled stream for alignment as a last resort for non-D405
                    try:
                        if pipeline_profile.get_stream(rs.stream.color):
                            align_to_stream = rs.stream.color
                        elif pipeline_profile.get_stream(rs.stream.depth):
                            align_to_stream = rs.stream.depth
                        elif pipeline_profile.get_stream(rs.stream.infrared):
                            align_to_stream = rs.stream.infrared
                    except RuntimeError: # If get_stream fails because it doesn't exist
                        pass # align_to_stream remains None or its previous value

            align = rs.align(align_to_stream) if align_to_stream else None
            if not align and (original_enable_depth_req or original_enable_color_req or original_enable_infrared_req): # Only warn if streams were expected
                print(f"[SingleRealsense {self.serial_number}] Warning: Alignment object could not be created. Alignment will be skipped.")

            # report global time
            sensor_to_set = None
            if product_name == "Intel RealSense D405":
                if original_enable_depth_req and device.first_depth_sensor():
                    sensor_to_set = device.first_depth_sensor() 
                elif (original_enable_color_req or original_enable_infrared_req) and device.first_infrared_sensor(): # D405 color uses IR
                    sensor_to_set = device.first_infrared_sensor()
            else: # For other D4xx cameras like D435
                if original_enable_color_req and device.first_color_sensor():
                    sensor_to_set = device.first_color_sensor()
                elif original_enable_depth_req and device.first_depth_sensor(): # Fallback if color not enabled/found
                    sensor_to_set = device.first_depth_sensor()
            
            if sensor_to_set:
                 sensor_to_set.set_option(rs.option.global_time_enabled, 1)
            # else:
                # print(f"[SingleRealsense {self.serial_number}] Warning: Could not get sensor to set global_time_enabled.")

            # setup advanced mode
            if self.advanced_mode_config is not None:
                json_text = json.dumps(self.advanced_mode_config)
                advanced_mode = rs.rs400_advanced_mode(device)
                advanced_mode.load_json(json_text)

            # get intrinsics
            stream_for_intrinsics = None
            # Priority: Color (if enabled), then Depth (if enabled), then Infrared (if enabled for D405 color/IR)
            if original_enable_color_req and not (product_name == "Intel RealSense D405"): # True color stream
                 try:
                    stream_for_intrinsics = pipeline_profile.get_stream(rs.stream.color)
                 except RuntimeError:
                    pass # print(f"[SingleRealsense {self.serial_number}] Warning: Color stream not found for intrinsics (non-D405).")
            
            if not stream_for_intrinsics and original_enable_depth_req:
                 try:
                    stream_for_intrinsics = pipeline_profile.get_stream(rs.stream.depth)
                 except RuntimeError:
                    pass # print(f"[SingleRealsense {self.serial_number}] Warning: Depth stream not found for intrinsics.")

            if not stream_for_intrinsics and (product_name == "Intel RealSense D405") and (original_enable_color_req or original_enable_infrared_req): # D405 uses IR for color
                 try:
                    stream_for_intrinsics = pipeline_profile.get_stream(rs.stream.infrared)
                 except RuntimeError:
                    pass # print(f"[SingleRealsense {self.serial_number}] Warning: Infrared stream not found for intrinsics (D405 color/IR mode).")
            
            if stream_for_intrinsics:
                intr = stream_for_intrinsics.as_video_stream_profile().get_intrinsics()
                order = ["fx", "fy", "ppx", "ppy", "height", "width"]
                for i, name in enumerate(order):
                    self.intrinsics_array.get()[i] = getattr(intr, name)
            else:
                 print(f"[SingleRealsense {self.serial_number}] Error: Could not get intrinsics from any enabled stream.")

            if original_enable_depth_req: # Check original request
                depth_sensor = device.first_depth_sensor()
                if depth_sensor:
                    depth_scale = depth_sensor.get_depth_scale()
                    self.intrinsics_array.get()[-1] = depth_scale
                # else:
                    # print(f"[SingleRealsense {self.serial_number}] Warning: Could not get depth sensor for depth_scale.")

            if self.verbose:
                print(f"[SingleRealsense {self.serial_number}] Main loop started.")

            # put frequency regulation
            put_idx = None
            put_start_time = self.put_start_time
            if put_start_time is None:
                put_start_time = time.time()

            iter_idx = 0
            t_start = time.time()

            # Create placeholders for data if streams are expected by ring buffer
            empty_color_placeholder = None
            if original_enable_color_req:
                empty_color_placeholder = np.zeros(self.resolution[::-1] + (3,), dtype=np.uint8)
            
            empty_ir_placeholder = None
            if original_enable_infrared_req:
                empty_ir_placeholder = np.zeros(self.resolution[::-1], dtype=np.uint8)

            with ThreadPoolExecutor(max_workers=2) as exec:
                while not self.stop_event.is_set():
                    try:
                        frameset = pipeline.wait_for_frames(timeout_ms=4000) 
                    except RuntimeError as e:
                        if "Frame didn't arrive within" in str(e):
                            print(f"[SingleRealsense {self.serial_number}] Warning: Frame timeout. Skipping frame acquisition for this cycle.")
                            continue 
                        else:
                            print(f"[SingleRealsense {self.serial_number}] RuntimeError in wait_for_frames: {e}")
                            raise 
                            
                    if not frameset:
                        # print(f"[SingleRealsense {self.serial_number}] Warning: Empty frameset received.")
                        continue
                        
                    receive_time = time.time()
                    
                    if align:
                        try:
                            frameset = align.process(frameset)
                        except Exception as e:
                            print(f"[SingleRealsense {self.serial_number}] Error during align.process: {e}. Using unaligned frameset.")
                            # frameset remains unaligned
                    # grab data
                    data = dict()
                    data["camera_receive_timestamp"] = receive_time
                    data["camera_capture_timestamp"] = frameset.get_timestamp() / 1000 # ms to s

                    # Initialize with placeholders if streams were originally requested
                    if original_enable_color_req:
                        data["color"] = empty_color_placeholder
                    if original_enable_infrared_req:
                        data["infrared"] = empty_ir_placeholder
                    
                    color_frame_data = None # Temp variable to hold actual color data if found
                    if self.enable_color: # This is the instance flag, possibly modified for D405 IR-as-color
                        try:
                            color_frame = frameset.get_color_frame()
                            if color_frame:
                                color_frame_data = np.asarray(color_frame.get_data())
                                data["camera_capture_timestamp"] = color_frame.get_timestamp() / 1000
                            elif product_name == "Intel RealSense D405" and self.enable_infrared: # D405 IR-as-color fallback
                                infrared_frame_for_color = frameset.get_infrared_frame()
                                if infrared_frame_for_color:
                                    ir_data = np.asarray(infrared_frame_for_color.get_data())
                                    color_frame_data = np.repeat(ir_data[..., np.newaxis], 3, axis=2)
                                    data["camera_capture_timestamp"] = infrared_frame_for_color.get_timestamp() / 1000 
                                # else:
                                    # if self.verbose and original_enable_color_req:
                                        # print(f"[SingleRealsense {self.serial_number}] Warning: D405 IR fallback for color: No IR frame.")
                            elif original_enable_color_req: 
                                 if self.verbose:
                                    print(f"[SingleRealsense {self.serial_number}] Warning: No color frame, but color was expected.")

                            if color_frame_data is not None and original_enable_color_req:
                                data["color"] = color_frame_data 
                            elif color_frame_data is None and original_enable_color_req:
                                if self.verbose: print(f"[SingleRealsense {self.serial_number}] Using placeholder for color frame.")

                        except RuntimeError as e:
                            if original_enable_color_req: 
                                print(f"[SingleRealsense {self.serial_number}] Error getting color frame: {e}. Placeholder will be used if color expected.")
                    
                    if self.enable_depth: 
                        depth_frame = frameset.get_depth_frame()
                        if depth_frame:
                            data["depth"] = np.asarray(depth_frame.get_data())
                        elif original_enable_depth_req: 
                            if "depth" not in data: 
                                data["depth"] = np.zeros(self.resolution[::-1], dtype=np.uint16)
                                if self.verbose: print(f"[SingleRealsense {self.serial_number}] Warning: No depth frame, using placeholder.")

                    if self.enable_infrared: 
                        infrared_frame_direct = frameset.get_infrared_frame()
                        if infrared_frame_direct:
                            if original_enable_infrared_req: 
                                data["infrared"] = np.asarray(infrared_frame_direct.get_data()) 
                        elif original_enable_infrared_req: 
                            if self.verbose: print(f"[SingleRealsense {self.serial_number}] Using placeholder for direct infrared frame.")

                    # apply transform
                    put_data = data.copy() # Use a copy to avoid modifying original data
                    if self.transform is not None:
                        put_data = self.transform(dict(data))

                    if self.put_downsample:
                        # put frequency regulation
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

                    # signal ready
                    if iter_idx == 0:
                        self.ready_event.set()

                    # put to vis
                    vis_data = data.copy()
                    if self.vis_transform == self.transform:
                        vis_data = put_data 
                    elif self.vis_transform is not None:
                        vis_data = self.vis_transform(dict(data))
                    
                    # Ensure vis_data always has expected keys if transforms might remove them
                    if original_enable_color_req and 'color' not in vis_data and empty_color_placeholder is not None:
                        vis_data['color'] = empty_color_placeholder
                    if original_enable_depth_req and 'depth' not in vis_data: # Assuming depth placeholder is zeros(shape, uint16)
                         vis_data['depth'] = np.zeros(self.resolution[::-1], dtype=np.uint16)
                    if original_enable_infrared_req and 'infrared' not in vis_data and empty_ir_placeholder is not None:
                         vis_data['infrared'] = empty_ir_placeholder

                    try:
                        self.vis_ring_buffer.put(vis_data, wait=False)
                    except KeyError as e:
                        print(f"[SingleRealsense {self.serial_number}] KeyError putting to vis_ring_buffer: {e}. Vis_data keys: {list(vis_data.keys())}")
                    except Exception as e: # Catch other potential errors during vis_ring_buffer put
                        print(f"[SingleRealsense {self.serial_number}] Error putting to vis_ring_buffer: {e}")

                    # record frame
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
                    elif self.video_recorder.is_ready() and original_enable_color_req and "color" not in rec_data: # Color was expected but not in rec_data
                         if self.verbose: print(f"[SingleRealsense {self.serial_number}] Warning: No color data in rec_data to record, but color was expected.")
                         # Optionally write placeholder if recorder expects frames always
                         # self.video_recorder.write_frame(empty_color_placeholder, frame_time=receive_time - self.record_start_time)

                    # perf
                    t_end = time.time()
                    duration = t_end - t_start
                    frequency = np.round(1 / duration, 1)
                    t_start = t_end
                    if self.verbose:
                        print(f"[SingleRealsense {self.serial_number}] FPS {frequency}")

                    # fetch command from queue
                    try:
                        commands = self.command_queue.get_all()
                        n_cmd = len(commands["cmd"])
                    except Empty:
                        n_cmd = 0

                    # execute commands
                    for i in range(n_cmd):
                        command = dict()
                        for key, value in commands.items():  # type: ignore
                            command[key] = value[i]
                        cmd = command["cmd"]
                        if cmd == Command.SET_COLOR_OPTION.value:
                            if product_name == "Intel RealSense D405":
                                continue # Skip for D405
                            sensor = device.first_color_sensor()
                            if sensor:
                                option = rs.option(command["option_enum"])
                                value = float(command["option_value"])
                                sensor.set_option(option, value)
                        elif cmd == Command.SET_DEPTH_OPTION.value:
                            sensor = device.first_depth_sensor()
                            if sensor:
                                option = rs.option(command["option_enum"])
                                value = float(command["option_value"])
                                sensor.set_option(option, value)
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
        finally:
            self.video_recorder.stop()
            # rs_config.disable_all_streams() # This might be problematic if pipeline is still active
            if 'pipeline' in locals() and pipeline:
                try:
                    pipeline.stop()
                except Exception as e:
                    print(f"[SingleRealsense {self.serial_number}] Error stopping pipeline: {e}")
            self.ready_event.set()

        if self.verbose:
            print(f"[SingleRealsense {self.serial_number}] Exiting worker process.")
