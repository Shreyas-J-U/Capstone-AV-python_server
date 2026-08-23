from environment.action import Action


class UERLEnvironment:

    def __init__(self, client):
        self.client = client

        self.current_episode = None
        self.current_step = None

    def reset(self):
        # TODO:
        # Send RESET to Unreal
        # Receive initial observation

        raise NotImplementedError

    def step(self, action: Action):
        action.validate()

        # TODO:
        # 1. Serialize action
        # 2. Send to Unreal
        # 3. Receive observation
        # 4. Deserialize observation
        # 5. Validate episode/step
        # 6. Return observation/reward/etc.

        raise NotImplementedError

    def close(self):
        self.client.close()