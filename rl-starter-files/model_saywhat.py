from model_babyai import *
from gymnasium.spaces.discrete import Discrete


class SayWhat_ACModel(nn.Module, torch_ac.RecurrentACModel):

    def __init__(self, obs_space, action_space, use_memory=True, use_text=True):
        super().__init__()

        self.action_space = Discrete(2)
        
        # self.action_model = action_model

        self.word_embedding_size = 128
        self.word_embedding = nn.Embedding(obs_space["text"], self.word_embedding_size)
        self.text_embedding_size = 128
        self.text_rnn = nn.GRU(self.word_embedding_size, self.text_embedding_size, batch_first=True)

        self.image_embedding_size = 128

        num_film_modules = 2
        self.film_layers = []
        for ni in range(num_film_modules):
            mod = FiLM(self.text_embedding_size, 128 if ni < num_film_modules-1 else self.image_embedding_size, 128, 128)
            self.film_layers.append(mod)
            self.add_module('FiLM_' + str(ni), mod)
        
        self.endpool = nn.MaxPool2d((7, 7), 2)

        self.memory_rnn = nn.LSTMCell(self.image_embedding_size, self.semi_memory_size)
        # Resize image embedding
        self.embedding_size = self.semi_memory_size

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, self.action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Initialize parameters correctly
        self.apply(init_params)

    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def forward(self, obs, memory, instr_embedding=None):
        embedding, memory = self._forward_without_heads(obs, memory, instr_embedding)
        dist, value = self._forward_through_heads(embedding)
        return dist, value, memory

    def _forward_without_heads(self, obs, memory, instr_embedding=None):
        if instr_embedding is None:
            instr_embedding = self._get_embed_text(obs.text)

        x = obs.image.transpose(1, 3).transpose(2, 3)
        with torch.no_grad():
            x = self.action_model.image_conv(x)
        
        for mod in self.film_layers:
            x = mod(x, instr_embedding)
        x = F.relu(self.endpool(x))
        x = x.reshape(x.shape[0], -1)
        
        hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
        hidden = self.memory_rnn(x, hidden)
        embedding = hidden[0]
        memory = torch.cat(hidden, dim=1)

        return embedding, memory
    
    def _forward_through_heads(self, embedding):
        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        return dist, value    


    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]


if __name__ == "__main__":
    from gymnasium.spaces.discrete import Discrete
    acmodel = ACModel({"image" : torch.Size([7, 7, 3]), "text" : 100}, Discrete(7), True, True)
    print(acmodel)
