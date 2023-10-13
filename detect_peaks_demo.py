import numpy as np
from lib.doa.detect_peaks import detect_peaks

x = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 200)) + np.random.randn(200) / 5
ind = detect_peaks(x, mph=0, mpd=20, show=True)

"""
この関数は、1次元のデータ配列からピーク（または谷）を検出するために使用されます。

* x: 1次元のデータ配列。
* mph: 最小ピーク高さ（Minimum Peak Height）。この値より大きいピークのみが検出されます。
* mpd: 最小ピーク距離（Minimum Peak Distance）。この値以上離れているピークのみが検出されます。
* threshold: 閾値。この値より大きいピーク（または小さい谷）のみが検出されます。
* edge: ピークのエッジをどのように扱うか（'rising', 'falling', 'both', None）。
* kpsh: 同じ高さのピークがmpdよりも近い場合にそれらを保持するかどうか。
* valley: 谷を検出する場合はTrue。
* show: 結果をプロットする場合はTrue。
* ax: matplotlibのAxesオブジェクト。指定された場合、そのAxesにプロットされます。

関数はピークのインデックスを1次元配列で返します。
"""