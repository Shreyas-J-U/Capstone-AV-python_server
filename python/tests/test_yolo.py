import unittest

# pyrefly: ignore [missing-import]
import numpy as np

from protocol.models import ImageData
from perception.yolo_detector import YOLODetector


class TestYOLODetector(unittest.TestCase):

    def test_raw_rgb_conversion(self):

        width = 640
        height = 480
        channels = 3

        rgb = np.zeros(
            (height, width, channels),
            dtype=np.uint8,
        )

        image = ImageData(
            width=width,
            height=height,
            channels=channels,
            data=rgb.tobytes(),
        )

        frame = YOLODetector.image_data_to_numpy(
            image
        )

        self.assertEqual(
            frame.shape,
            (480, 640, 3)
        )

        self.assertEqual(
            frame.dtype,
            np.uint8
        )

        self.assertEqual(
            len(frame.tobytes()),
            640 * 480 * 3
        )


if __name__ == "__main__":
    unittest.main()