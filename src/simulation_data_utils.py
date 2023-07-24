import wave

import numpy as np
import pyroomacoustics as pra


def normalize_and_pad_audio_files(wave_files):
    audio_data = []
    n_samples = 0
    n_sources = len(wave_files)

    for wave_file in wave_files:
        with wave.open(wave_file) as wav:
            data = wav.readframes(wav.getnframes())
            data = np.frombuffer(data, dtype=np.int16)
            n_samples = max(wav.getnframes(), n_samples)
            data = data / np.iinfo(np.int16).max
            audio_data.append(data)

    for s in range(n_sources):
        if len(audio_data[s]) < n_samples:
            pad_width = n_samples - len(audio_data[s])
            audio_data[s] = np.pad(audio_data[s], (0, pad_width), "constant")
        audio_data[s] /= np.std(audio_data[s])
    return audio_data


def create_outdoor_room(room_dim, fs):
    m = pra.make_materials(floor="rough_concrete")

    # Create a material for the air (other surfaces)
    air_absorption = 1.0
    air_material = pra.Material(air_absorption)

    # Set the air material to all other surfaces
    for direction in ["ceiling", "east", "west", "north", "south"]:
        m[direction] = air_material

    room = pra.ShoeBox(room_dim, fs=fs, materials=m, max_order=17)
    return room


# マイクロホンアレイまたは音源を円形に配置する関数
def circular_layout(center, radius, num_items):
    angles = np.linspace(0, 2 * np.pi, num_items, endpoint=False)
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    z = np.ones_like(x) * center[2]
    return np.vstack((x, y, z))


def perform_simulation(room):
    # 全ての音源とマイクロホンが部屋の中に入っていることを確認
    for source in room.sources:
        assert room.is_inside(source.position), "Some sources are outside the room."
    for mic in room.mic_array.R.T:
        assert room.is_inside(mic), "Some microphones are outside the room."

    print("All sources and microphones are inside the room.")

    room.simulate()
    simulated_signals = room.mic_array.signals
    return simulated_signals
