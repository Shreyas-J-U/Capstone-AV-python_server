# pyrefly: ignore [missing-import]
from perception.trajectory_predictor import TrajectoryPredictor


def main():
    print()
    print("=" * 60)
    print("TRAJECTORY PREDICTOR TEST")
    print("=" * 60)

    predictor = TrajectoryPredictor(
        prediction_horizon=5
    )

    history = [
        (100.0, 200.0),
        (105.0, 203.0),
        (110.0, 206.0),
        (115.0, 209.0),
        (120.0, 212.0),
    ]

    print()
    print("Observed trajectory:")
    print(history)

    predictions = predictor.predict(
        history
    )

    print()
    print("Predicted future trajectory:")

    for i, point in enumerate(
        predictions,
        start=1
    ):
        print(
            f"t+{i}: "
            f"({point[0]:.2f}, {point[1]:.2f})"
        )

    # ----------------------------------------
    # Assertions
    # ----------------------------------------

    assert len(predictions) == 5

    assert predictions[0] == (
        125.0,
        215.0,
    )

    assert predictions[-1] == (
        145.0,
        227.0,
    )

    print()
    print("=" * 60)
    print("TRAJECTORY PREDICTOR TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()