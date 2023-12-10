import copy
import torch
from torch import nn


class SequentialVAE:

    def __init__(self, char_length=27, n_characters=5, n_latents=4):
        self.char_length = char_length
        self.n_characters = n_characters
        self.encoder = Encoder(char_length, n_characters, n_latents)
        self.decoder = Decoder(char_length, n_characters, n_latents)

    def learn_from(self, obs):

        n_chars = int(obs.shape[0] / self.char_length)
        print(f"n_chars: {n_chars}")
        print(f"self.n_characters: {self.n_characters}")
        """
        if n_chars < self.n_characters:
            new_obs = torch.concat([obs, torch.zeros((self.n_characters - n_chars) * self.char_length)])
        else:
            new_obs = obs

        mean, log_var, size = self.encoder(new_obs)
        print(mean)
        print(log_var)
        print(size)
        print("=" * 70)

        predicted_obs = self.decoder(mean)
        print(predicted_obs)
        print(predicted_obs.shape)
        print("=" * 70)
        """

        vfe = RecursiveVFE(self.encoder, self.decoder, self.char_length, self.n_characters).compute(obs)
        print(vfe)
        print("=" * 70)


class RecursiveVFE:

    def __init__(self, encoder, decoder, char_length=27, n_characters=5):
        self.n_leaf_nodes = 0
        self.encoder = encoder
        self.decoder = decoder
        self.char_length = char_length
        self.n_characters = n_characters
        self.min_vfe = None
        self.min_sizes = None

    def compute(self, obs, vfe=0, sizes=None):

        # Compute number of characters left.
        n_chars = int(obs.shape[0] / self.char_length)

        # If no more character left, save the smallest VFE and return the VFE.
        if n_chars == 0:
            if self.min_vfe is None or vfe < self.min_vfe:
                self.min_vfe = vfe
                self.min_sizes = sizes
            self.n_leaf_nodes += 1
            print(self.n_leaf_nodes, sizes)
            return self.min_vfe

        for i in range(min(n_chars, self.n_characters)):

            # Create an observation containing n characters, padding with zeros if necessary.
            if n_chars >= self.n_characters:
                x = obs[:self.n_characters * self.char_length]
            else:
                x = torch.concat([obs, torch.zeros((self.n_characters - n_chars) * self.char_length)])

            # Update the VFE and sizes.
            new_vfe = vfe + self.compute_partial_vfe(x)
            if sizes is None:
                new_sizes = [i + 1]
            else:
                new_sizes = sizes + [i + 1]

            # Create the new observations and call the compute function recursively.
            if i + 1 <= n_chars:
                new_obs = obs[(i + 1) * self.char_length:]
            else:
                new_obs = torch.zeros(0)

            self.compute(new_obs, new_vfe, new_sizes)

        return self.min_vfe

    def compute_partial_vfe(self, x):
        return 1  # TODO


class Encoder:

    def __init__(self, char_length=27, n_characters=5, n_latents=5):
        self.char_length = char_length
        self.n_characters = n_characters
        self.n_latents = n_latents

        n_features = 10
        self.character_network = nn.Sequential(
            nn.Linear(char_length, n_features),
            nn.ReLU()
        )
        self.linear_network = nn.Sequential(
            nn.Linear(n_features * n_characters, 100),
            nn.ReLU(),
            nn.Linear(100, 2 * n_latents + n_characters),
        )

    def __call__(self, obs):

        # Apply the character network on each character.
        char_embeddings = []
        for i in range(self.n_characters):
            char = obs[i * self.char_length:(i + 1) * self.char_length]
            char_embedding = self.character_network(char)
            char_embeddings.append(char_embedding)

        # Compute the encoder output from character embedding.
        char_embeddings = torch.concat(char_embeddings)
        output = self.linear_network(char_embeddings)
        return output[0:self.n_latents], output[self.n_latents:2 * self.n_latents], output[2 * self.n_latents:]


class Decoder:

    def __init__(self, char_length=27, n_characters=5, n_latents=5):
        self.char_length = char_length
        self.n_characters = n_characters
        self.n_latents = n_latents

        self.n_features = 10
        self.linear_network = nn.Sequential(
            nn.Linear(n_latents, 100),
            nn.ReLU(),
            nn.Linear(100, self.n_features * n_characters),
        )
        self.feature_network = nn.Sequential(
            nn.Linear(self.n_features, char_length),
            nn.ReLU()
        )

    def __call__(self, x):

        # Compute all the features.
        all_features = self.linear_network(x)

        # Apply the feature network on each set of features.
        chars = []
        for i in range(self.n_characters):
            features = all_features[i * self.n_features:(i + 1) * self.n_features]
            char = self.feature_network(features)
            chars.append(char)

        return torch.concat(chars)
