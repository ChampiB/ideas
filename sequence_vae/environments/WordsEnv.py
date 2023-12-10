from gym import Env
import numpy as np
from torch.nn.functional import one_hot
import torch


class WordsEnv(Env):

    def __init__(self, words=None, n_words=3):
        """
        Create an environment generating sequences of words
        :param words: the list of valid words
        :param n_words: the number of words in each sequence
        """
        self.alphabet = " abcdefghijklmnopqrstuvwxyz"
        self.words = ["one", "two", "three"] if words is None else words
        self.n_words = n_words

    def step(self, action):
        """
        Perform on step in the environment
        :param action: the action to perform
        :return: a tuple of the form (observation, reward, done, additional_information)
        """
        return self.generate_next_observation(), 0, False, {}

    def reset(self):
        """
        Reset the environment
        :return: the initial observation
        """

        return self.generate_next_observation()

    def generate_next_observation(self):
        """
        Generate the next observation that the agent will receive
        :return: the next observation
        """

        # Create a string containing a list of words.
        word_indices = np.random.choice(len(self.words), self.n_words)
        text = ""
        for i, word_index in enumerate(word_indices):
            if i != 0:
                text += " "
            text += self.words[word_index]

        # Create a bit representation of the string s.t. each character is one-hot encoding of 27 characters.
        obs = torch.zeros(len(text), dtype=torch.int64)
        for i, character in enumerate(text):
            obs[i] = self.alphabet.index(character)
        obs = one_hot(obs, len(self.alphabet))
        return torch.concat([o for o in obs]).to(torch.float32)
