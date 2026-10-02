"""TDC trigger packets must decode to the trigger time within the fine step
(25/4096 ns x 512/12, about 0.26 ns), including float pulse times a hair below
a 25 ns boundary, which used to come out 25 ns early (fine part wrapped to 0
without carrying into the coarse count)."""
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from lumacam.optics import encode_tdc_packet


def decode_tdc_ns(packet):
    """The reader's decoding (C++ decoder quoted in encode_tdc_packet)."""
    temp = struct.unpack('<Q', packet)[0]
    coarse = (temp >> 12) & 0xFFFFFFFF
    tmpfine = (temp >> 5) & 0xF
    fine = (temp & 0x0E00) | (((tmpfine - 1) << 9) // 12)
    return coarse * 25.0 + fine * 25.0 / 4096.0


def test_round_trip():
    rng = np.random.default_rng(1)
    pulses = np.arange(1, 400) * 1000.0
    times = np.r_[pulses, pulses - 1e-7, pulses + 1e-7, rng.uniform(0, 1e9, 2000)]
    err = np.array([decode_tdc_ns(encode_tdc_packet(t, 1)) - t for t in times])
    assert np.abs(err).max() < 0.3, (times[np.abs(err).argmax()], err[np.abs(err).argmax()])


if __name__ == '__main__':
    test_round_trip()
    print('ok')
