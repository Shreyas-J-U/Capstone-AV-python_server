import threading
import time
import unittest

from transport.tcp_server import TCPServer
from protocol.connection import ProtocolConnection
from environment.ue_environment import UERLEnvironment
from agents.test_agent import MyAgent
from tests.fake_unreal_episodes import main as run_fake_unreal


class TestMultiEpisodeLoop(unittest.TestCase):

    def test_end_to_end_multi_episode_loop(self):
        host = "127.0.0.1"
        port = 9876
        num_episodes = 3
        steps_per_episode = 4

        server_errors = []
        client_errors = []

        def server_thread_func():
            try:
                server = TCPServer(host=host, port=port)
                server.start()
                server.accept()

                connection = ProtocolConnection(server)
                env = UERLEnvironment(connection)
                env.perform_handshake()

                agent = MyAgent()

                for ep in range(1, num_episodes + 1):
                    obs = env.reset()
                    self.assertEqual(obs.episode_id, ep)
                    self.assertEqual(obs.step_id, 0)

                    terminated = False
                    steps = 0

                    while not terminated:
                        action = agent.act(obs)
                        next_obs, reward, terminated, info = env.step(action)
                        steps += 1
                        obs = next_obs

                    self.assertEqual(steps, steps_per_episode)
                    self.assertTrue(terminated)

                env.close()
                server.close()
            except Exception as e:
                server_errors.append(e)

        def client_thread_func():
            try:
                # Small delay to ensure server socket is listening
                time.sleep(0.2)
                import sys
                old_argv = sys.argv
                sys.argv = [
                    "fake_unreal_episodes.py",
                    "--host", host,
                    "--port", str(port),
                    "--episodes", str(num_episodes),
                    "--steps-per-episode", str(steps_per_episode),
                ]
                try:
                    run_fake_unreal()
                finally:
                    sys.argv = old_argv
            except Exception as e:
                client_errors.append(e)

        server_thread = threading.Thread(target=server_thread_func)
        client_thread = threading.Thread(target=client_thread_func)

        server_thread.start()
        client_thread.start()

        server_thread.join(timeout=10.0)
        client_thread.join(timeout=10.0)

        self.assertFalse(server_thread.is_alive(), "Server thread timed out!")
        self.assertFalse(client_thread.is_alive(), "Client thread timed out!")

        if server_errors:
            self.fail(f"Server thread raised error: {server_errors[0]}")
        if client_errors:
            self.fail(f"Client thread raised error: {client_errors[0]}")


if __name__ == "__main__":
    unittest.main()
