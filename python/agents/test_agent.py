from protocol.models import Action, Observation


class MyAgent:
    """
    Temporary fake RL agent.

    This is NOT the real RL model.
    It is only used to verify the Python
    infrastructure and protocol pipeline.
    """

    def act(self, observation: Observation) -> Action:

        print()
        print("========== AGENT ACT ==========")

        print(
            f"Episode : {observation.episode_id}"
        )

        print(
            f"Step    : {observation.step_id}"
        )

        print(
            f"Frame   : {observation.frame_id}"
        )

        print(
            f"Reward  : {observation.reward}"
        )

        print(
            f"Sensors: {len(observation.sensors)}"
        )

        # Temporary deterministic action.
        action = Action(
            episode_id=observation.episode_id,
            step_id=observation.step_id,
            throttle=0.5,
            steering=0.0,
            brake=False,
        )

        print()
        print("Action generated:")
        print(
            f"  throttle = {action.throttle}"
        )
        print(
            f"  steering = {action.steering}"
        )
        print(
            f"  brake    = {action.brake}"
        )

        print("===============================")

        return action

    def train(self, transition):
        """
        Placeholder for the actual RL training logic.

        The real agent will eventually receive:

            observation
            action
            reward
            next_observation
            terminated
        """

        pass