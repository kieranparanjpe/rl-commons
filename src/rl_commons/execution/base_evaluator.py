from __future__ import annotations

import queue
import threading
from abc import ABC, abstractmethod

import torch

from rl_commons.mdp import MdpGym, MdpConfig


class BaseEvaluator(ABC):

    def __init__(self, environment_id: str, mdp_config: MdpConfig = MdpConfig(), **mdp_kwargs):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._stop = threading.Event()
        self._mdp = MdpGym(environment_id, self.device,
                           render_mode='human', mdp_config=mdp_config, **mdp_kwargs)

    @staticmethod
    def load_checkpoint(path: str) -> dict:
        """Load a torch checkpoint from path, returning an empty dict if path is falsy."""
        return torch.load(path, weights_only=True) if path else {}

    def _listen_for_commands(self):
        commands = queue.Queue()
        while not self._stop.is_set():
            cmd = input().strip().lower()
            commands.put(cmd)
            if cmd in {"x", "close", "quit", "exit"}:
                self._stop.set()

    def evaluate(self):
        """Template method: starts the keyboard listener, runs _run(), then closes the MDP."""
        print("Type 'x' or 'close' + Enter to stop.")
        threading.Thread(target=self._listen_for_commands, daemon=True).start()
        self._run()
        self._mdp.close()

    @abstractmethod
    def _run(self):
        """The inner evaluation loop. Runs until self._stop is set."""
        pass
