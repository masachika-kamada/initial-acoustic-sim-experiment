import numpy as np
import pyroomacoustics as pra

from src.file_io import load_signal_from_wav, write_signal_to_wav
from src.simulation_data_utils import circular_layout, create_outdoor_room, perform_simulation
from src.visualization_tools import play_audio, plot_room, plot_room_views

np.random.seed(0)


class Drone:
    def __init__(self,
                 prop_height: float,     prop_radius: float, prop_num: int, source_path: str,
                 mic_offset: np.ndarray, mic_radius : float, mic_num : int):
        # ドローン
        self.prop_height = prop_height
        self.prop_radius = prop_radius
        self.prop_num = prop_num
        self.source_path = source_path
        # マイク
        self.mic_offset = mic_offset
        self.mic_radius = mic_radius
        self.mic_num = mic_num

    def place(self, room: pra.ShoeBox):
        self.place_propeller(room)
        self.place_mic_array(room)

    def place_propeller(self, room: pra.ShoeBox):
        self.prop_center = np.array([room.shoebox_dim[0] / 2, room.shoebox_dim[1] / 2, self.prop_height])
        source_signal = load_signal_from_wav(self.source_path, room.fs)
        prop_pos = circular_layout(self.prop_center, self.prop_radius, self.prop_num)
        # 音源を部屋に追加
        samples_per_source = len(source_signal) // self.prop_num
        for i, pos in enumerate(prop_pos.T):
            room.add_source(pos, signal=source_signal[samples_per_source * i:samples_per_source * (i + 1)])

    def place_mic_array(self, room: pra.ShoeBox):
        mic_pos = circular_layout(self.prop_center + self.mic_offset, self.mic_radius, self.mic_num)
        mic_array = pra.MicrophoneArray(mic_pos, room.fs)
        room.add_microphone_array(mic_array)


class Speakers:
    def __init__(self, poss: np.ndarray, source_paths: list):
        self.poss = poss
        self.source_paths = source_paths

    def place(self, room: pra.ShoeBox):
        for pos, path in zip(self.poss, self.source_paths):
            source_signal = load_signal_from_wav(path, room.fs)
            room.add_source(pos, signal=source_signal)


class Simulator:
    def __init__(self, room_dim: np.ndarray, fs: int):
        self.room_noise = create_outdoor_room(room_dim, fs)
        self.room = create_outdoor_room(room_dim, fs)

    def simulate(self, drone: Drone, speakers: Speakers, save_dir: str):
        drone.place(self.room_noise)
        simulated_signals = perform_simulation(self.room_noise)
        print(simulated_signals.shape)
        write_signal_to_wav(simulated_signals, f"{save_dir}/1.wav", self.room_noise.fs)

        drone.place(self.room)
        speakers.place(self.room)
        plot_room(self.room)
        simulated_signals = perform_simulation(self.room)
        print(simulated_signals.shape)
        write_signal_to_wav(simulated_signals, f"{save_dir}/2.wav", self.room.fs)


if __name__ == "__main__":
    drone = Drone(prop_height=15, prop_radius=0.2, prop_num=4,
                  source_path="./data/processed/propeller/p2000_2/dst.wav",
                  mic_offset=np.array([0, 0, -5]), mic_radius=0.05, mic_num=8)

    speakers = Speakers(poss=np.array([[35, 25, 0], [15, 20, 0]]),
                        source_paths=["./data/raw/sample/arctic_a0001.wav",
                                      "./data/raw/sample/arctic_a0002.wav"])

    simulator = Simulator(room_dim=np.array([50, 50, 50]), fs=16000)
    simulator.simulate(drone, speakers, "./data/simulation/trial1")
