from abc import ABC, abstractmethod


class RLAgent(ABC):

    @abstractmethod
    def act(self, observation):
        raise NotImplementedError

    @abstractmethod
    def train(self, transition):
        raise NotImplementedError


