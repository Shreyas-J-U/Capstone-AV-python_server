import argparse

from transport.tcp_server import TCPServer
from protocol.connection import ProtocolConnection
from environment.ue_environment import UERLEnvironment
from agents.test_agent import MyAgent


def main():
    parser = argparse.ArgumentParser(description="RL TCP Server for Unreal Engine 5")
    parser.add_argument("--host", default="127.0.0.1", help="Binding host address")
    parser.add_argument("--port", type=int, required=True, help="Listening TCP port")
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes to execute with connected Unreal client",
    )
    args = parser.parse_args()

    server = TCPServer(host=args.host, port=args.port)

    try:
        server.start()
        print(f"Python RL Server listening on {args.host}:{args.port}...")
        server.accept()
        print("Unreal client connected successfully.")

        connection = ProtocolConnection(server)
        env = UERLEnvironment(connection)

        print("Executing HELLO / HELLO_ACK protocol handshake...")
        env.perform_handshake()
        print("Handshake OK!")

        agent = MyAgent()

        for ep in range(1, args.episodes + 1):
            print(f"\n=================== EPISODE {ep} START ===================")
            
            # 1. Reset environment for new episode
            obs = env.reset()
            print(
                f"[Server] Initial Obs received: ep={obs.episode_id}, step={obs.step_id}, "
                f"sim_time={obs.simulation_time:.2f}s"
            )

            terminated = False
            total_reward = obs.reward
            step_count = 0

            # 2. Episode step loop
            while not terminated:
                action = agent.act(obs)

                next_obs, reward, terminated, info = env.step(action)
                step_count += 1
                total_reward += reward

                print(
                    f"[Server] Step {next_obs.step_id} executed: "
                    f"reward={reward:.2f}, terminated={terminated}, info={info}"
                )

                # Train agent on transition
                transition = {
                    "obs": obs,
                    "action": action,
                    "reward": reward,
                    "next_obs": next_obs,
                    "terminated": terminated,
                }
                agent.train(transition)

                obs = next_obs

            print(
                f"=================== EPISODE {ep} END ===================\n"
                f"Total Steps: {step_count} | Total Episode Reward: {total_reward:.2f}\n"
            )

        print("All target episodes completed successfully!")

    except KeyboardInterrupt:
        print("\nStopping RL server manually...")
    except Exception as e:
        print(f"Error during execution: {type(e).__name__}: {e}")
        raise
    finally:
        if 'env' in locals():
            env.close()
        server.close()
        print("Server closed.")


if __name__ == "__main__":
    main()