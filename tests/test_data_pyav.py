from pathlib import Path

import numpy as np
import pytest

av = pytest.importorskip("av", reason="PyAV is an optional preprocessing dependency")

from genet.data.video import decode_video


def test_pyav_encoded_video_decode(tmp_path: Path):
    path = tmp_path / "tiny.mp4"
    try:
        with av.open(str(path), mode="w") as container:
            stream = container.add_stream("mpeg4", rate=5)
            stream.width = 16
            stream.height = 16
            stream.pix_fmt = "yuv420p"
            for index in range(3):
                image = np.full((16, 16, 3), index * 40, dtype=np.uint8)
                frame = av.VideoFrame.from_ndarray(image, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
    except Exception as exc:
        pytest.skip(f"local FFmpeg build cannot encode the test fixture: {exc}")
    decoded = decode_video(path)
    assert decoded.frames.shape == (3, 16, 16, 3)
    assert len(decoded.timestamps) == 3
