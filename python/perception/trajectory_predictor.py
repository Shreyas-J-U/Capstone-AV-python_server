from typing import List, Tuple


Point = Tuple[float, float]


class TrajectoryPredictor:
    """
    Baseline trajectory predictor.

    Uses the recent movement of an object to estimate
    its future positions using a constant-velocity model.

    This is a baseline implementation.

    Later, this component can be replaced by Trajectron++
    without changing the perception or planning interfaces.
    """

    def __init__(
        self,
        prediction_horizon: int = 10,
    ):
        """
        Initialize the trajectory predictor.

        Args:
            prediction_horizon:
                Default number of future positions to predict.
        """

        if prediction_horizon <= 0:
            raise ValueError(
                "prediction_horizon must be greater than 0."
            )

        self.prediction_horizon = prediction_horizon

    def predict(
        self,
        history: List[Point],
        steps: int | None = None,
    ) -> List[Point]:
        """
        Predict future positions from observed history.

        Uses a simple constant-velocity model based on
        the displacement between the last two observations.

        Args:
            history:
                List of observed (x, y) positions.

            steps:
                Number of future positions to predict.
                If omitted, self.prediction_horizon is used.

        Returns:
            List of predicted future (x, y) positions.

        Example:

            history = [
                (100.0, 200.0),
                (105.0, 203.0),
                (110.0, 206.0),
            ]

            predictor.predict(history, steps=3)

            returns:

            [
                (115.0, 209.0),
                (120.0, 212.0),
                (125.0, 215.0),
            ]
        """

        # --------------------------------------------------
        # Validate history
        # --------------------------------------------------

        if len(history) < 2:
            return []

        # --------------------------------------------------
        # Determine prediction horizon
        # --------------------------------------------------

        if steps is None:
            steps = self.prediction_horizon

        if steps <= 0:
            raise ValueError(
                "steps must be greater than 0."
            )

        # --------------------------------------------------
        # Last two observed positions
        # --------------------------------------------------

        x1, y1 = history[-2]
        x2, y2 = history[-1]

        # --------------------------------------------------
        # Estimate velocity
        # --------------------------------------------------

        vx = x2 - x1
        vy = y2 - y1

        # --------------------------------------------------
        # Generate future positions
        # --------------------------------------------------

        predictions: List[Point] = []

        current_x = x2
        current_y = y2

        for _ in range(steps):

            current_x += vx
            current_y += vy

            predictions.append(
                (
                    current_x,
                    current_y,
                )
            )

        return predictions