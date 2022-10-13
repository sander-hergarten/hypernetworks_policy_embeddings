import numpy as np
from cv2 import VideoWriter_fourcc, VideoWriter
from dataclasses import dataclass, field
from ..utils import generate_filename


@dataclass
class FrameStorage:
    width: int
    height: int

    metadata: str

    frame_rate: int = 15
    frame_list: list = field(default_factory=list)

    @property
    def video_length(self) -> int:
        return len(self.frame_list) // self.frame_rate

    def add_frame(self, rgb_array):
        self.frame_list.append(rgb_array)

    def __next__(self):
        temp_frame_list = self.frame_list
        while len(temp_frame_list) != 0:
            yield temp_frame_list.pop(0)

    def __iter__(self):
        return next(self)


@dataclass
class VideoStorage:
    # TODO: switch to lookupable object. interact with it more seamlessly
    _video_dict: dict[str, FrameStorage] = field(default_factory=dict)

    def add_frame_storage(self, identifier: str, frame_storage: FrameStorage):
        self._video_dict[identifier] = frame_storage

    def to_video(self, identifier: str, filename: str):
        frame_storage = self._video_dict[identifier]

        fourcc = VideoWriter_fourcc(*"MP42")
        video = VideoWriter(
            f"{filename}.mp4",
            fourcc,
            float(frame_storage.frame_rate),
            (frame_storage.width, frame_storage.height),
        )

        for frame in frame_storage:
            video.write(frame)
        video.release()

    def all_to_video(self, relative_path: str) -> list[str]:
        paths = []
        for identifier, frame_storage in self._video_dict.items():
            paths.append(generate_filename(relative_path))
            self.to_video(identifier, paths[-1])

        return paths

    def empty_storage(self, relative_path: str) -> list[str]:
        paths = self.all_to_video(relative_path)
        self._video_dict = {}

        return paths


class VideoTracker:
    # TODO: add support for in memory and out of memory video storage
    render_env_is_active: bool = False

    def __init__(self, cycle_length: int):
        self.cycle_length = cycle_length
        self.video_storage = VideoStorage()

    def reset(self, **environment_interface):
        self.current_episode = environment_interface["get_current_episode"]()

        episode_offset = self.current_episode % self.cycle_length

        if episode_offset == 0:
            self.current_frame_storage = FrameStorage(512, 512, "")
            environment_interface["switch_to_render_env"]()
            self.render_env_is_active = True

        elif episode_offset == 1:
            environment_interface["switch_to_default_env"]()
            self.render_env_is_active = False

    def __call__(self, step_data, **environment_interface):
        if self.render_env_is_active:
            self.current_frame_storage.add_frame(step_data[-1]["rgb"])
