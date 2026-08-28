from pathlib import Path

from video.video_sampler import sample_video


def main():

    base_dir = Path(__file__).resolve().parent.parent

    video_path = (
        base_dir
        / "assets"
        / "test_video.mp4"
    )

    output_dir = (
        base_dir
        / "tests"
        / "assets"
        / "sampled_frames"
    )

    count = sample_video(
        video_path=str(video_path),
        output_dir=str(output_dir),
        sample_fps=5.0,
        max_frames=50,
    )

    assert count > 0

    print()
    print(
        f"TEST PASSED: "
        f"{count} frames extracted."
    )


if __name__ == "__main__":
    main()