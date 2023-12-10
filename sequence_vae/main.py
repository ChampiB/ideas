from sequence_vae.agents.SquentialVAE import SequentialVAE
from sequence_vae.environments.WordsEnv import WordsEnv


def hello_world():
    # TODO for i, l in enumerate(" abcdefghijklmnopqrstuvwxyz"):
    # TODO     print(f"{i}: {l}")

    n_iterations = 1

    agent = SequentialVAE()

    env = WordsEnv()
    obs = env.reset()
    for i in range(n_iterations):
        agent.learn_from(obs)
        obs, reward, done, info = env.step(0)


if __name__ == '__main__':
    hello_world()
