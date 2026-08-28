from pathlib import Path
from typing import Optional

# pyrefly: ignore [missing-import]
import cv2


def sample_video(
    video_path: str,
    output_dir: str,
    sample_fps: float = 5.0,
    max_frames: Optional[int] = 50,
) -> int:
    """
    Extract temporally ordered frames from a video.

    Parameters
    ----------
    video_path:
        Path to the input video.

    output_dir:
        Directory where sampled frames will be saved.

    sample_fps:
        Number of frames to extract per second.

    max_frames:
        Maximum number of frames to save.
        None means no limit.

    Returns
    -------
    int
        Number of frames successfully extracted.
    """

    video_path = Path(video_path)
    output_dir = Path(output_dir)

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    if sample_fps <= 0:
        raise ValueError(
            "sample_fps must be greater than 0"
        )

    if max_frames is not None and max_frames <= 0:
        raise ValueError(
            "max_frames must be greater than 0 "
            "or None"
        )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Open video
    # --------------------------------------------------------

    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Unable to open video: {video_path}"
        )

    # --------------------------------------------------------
    # Read video properties
    # --------------------------------------------------------

    original_fps = capture.get(
        cv2.CAP_PROP_FPS
    )

    total_frames = int(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    duration = (
        total_frames / original_fps
        if original_fps > 0
        else 0
    )

    print()
    print("=" * 60)
    print("VIDEO SAMPLER")
    print("=" * 60)

    print(
        f"Input video   : {video_path}"
    )

    print(
        f"Original FPS  : {original_fps:.2f}"
    )

    print(
        f"Total frames  : {total_frames}"
    )

    print(
        f"Duration      : {duration:.2f} seconds"
    )

    print(
        f"Sample FPS    : {sample_fps}"
    )

    print(
        f"Maximum frames: {max_frames}"
    )

    print(
        f"Output        : {output_dir}"
    )

    # --------------------------------------------------------
    # Calculate frame interval
    # --------------------------------------------------------

    frame_interval = max(
        1,
        round(
            original_fps / sample_fps
        ),
    )

    actual_sample_fps = (
        original_fps / frame_interval
        if original_fps > 0
        else sample_fps
    )

    print(
        f"Frame interval: every "
        f"{frame_interval} original frame(s)"
    )

    print(
        f"Actual sample rate: "
        f"{actual_sample_fps:.2f} FPS"
    )

    print()
    print("Extracting frames...")
    print()

    # --------------------------------------------------------
    # Extract
    # --------------------------------------------------------

    frame_index = 0
    saved_count = 0

    try:

        while True:

            success, frame = capture.read()

            if not success:
                break

            # Only keep selected frames.
            if frame_index % frame_interval == 0:

                output_path = (
                    output_dir
                    / f"frame_{saved_count + 1:06d}.png"
                )

                write_success = cv2.imwrite(
                    str(output_path),
                    frame,
                )

                if not write_success:
                    raise RuntimeError(
                        f"Failed to write frame: "
                        f"{output_path}"
                    )

                saved_count += 1

                print(
                    f"Saved frame "
                    f"{saved_count:03d}: "
                    f"{output_path.name}"
                )

                if (
                    max_frames is not None
                    and saved_count >= max_frames
                ):
                    break

            frame_index += 1

    finally:

        capture.release()

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("VIDEO SAMPLING COMPLETE")
    print("=" * 60)

    print(
        f"Frames read   : {frame_index + 1}"
    )

    print(
        f"Frames saved  : {saved_count}"
    )

    print(
        f"Output folder : {output_dir}"
    )

    print("=" * 60)

    return saved_count