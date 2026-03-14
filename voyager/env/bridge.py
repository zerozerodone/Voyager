import os
import os.path
import secrets
import shutil
import time
import warnings
from typing import SupportsFloat, Any, Tuple, Dict

import requests
import json

import gymnasium as gym
from gymnasium.core import ObsType

import voyager.utils as U

from .minecraft_launcher import MinecraftInstance
from .process_monitor import SubprocessMonitor

_NODE22_SEARCH_PATHS = [
    os.path.expanduser("~/.local/node22/bin/node"),
    os.path.expanduser("~/.local/node20/bin/node"),
]

_CODE_BLOCKLIST = [
    "child_process",
    "process.exit",
    "process.env",
    "process.kill",
    "require('fs')",
    'require("fs")',
    "require('net')",
    'require("net")',
    "require('http')",
    'require("http")',
    "require('https')",
    'require("https")',
    "require('os')",
    'require("os")',
    "require('dgram')",
    'require("dgram")',
    "import(",
]


def _find_node_binary():
    for path in _NODE22_SEARCH_PATHS:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    system_node = shutil.which("node")
    if system_node:
        return system_node
    raise FileNotFoundError(
        "No Node.js binary found. Install Node 22 LTS to ~/.local/node22/"
    )


def validate_code(code: str) -> None:
    for pattern in _CODE_BLOCKLIST:
        if pattern in code:
            raise ValueError(
                f"[Sandbox] Code rejected: contains blocked pattern '{pattern}'"
            )


class VoyagerEnv(gym.Env):
    def __init__(
        self,
        mc_port=None,
        azure_login=None,
        server_host="http://127.0.0.1",
        server_port=3000,
        request_timeout=600,
        log_path="./logs",
        pause_between_steps=True,
        bot_username="bot",
        bot_skin=None,
    ):
        if not mc_port and not azure_login:
            raise ValueError("Either mc_port or azure_login must be specified")
        if mc_port and azure_login:
            warnings.warn(
                "Both mc_port and mc_login are specified, mc_port will be ignored"
            )
        self.mc_port = mc_port
        self.azure_login = azure_login
        self.server = f"{server_host}:{server_port}"
        self.server_port = server_port
        self.request_timeout = request_timeout
        self.log_path = log_path
        self.bot_username = bot_username
        self.bot_skin = bot_skin

        self.auth_token = secrets.token_hex(32)
        self._auth_headers = {"Authorization": f"Bearer {self.auth_token}"}
        os.environ["VOYAGER_AUTH_TOKEN"] = self.auth_token

        self.mineflayer = self.get_mineflayer_process(server_port)
        if azure_login:
            self.mc_instance = self.get_mc_instance()
        else:
            self.mc_instance = None
        self.has_reset = False
        self.reset_options = None
        self.connected = False
        self.server_paused = False
        self.pause_between_steps = pause_between_steps

    def get_mineflayer_process(self, server_port):
        U.f_mkdir(self.log_path, "mineflayer")
        file_path = os.path.abspath(os.path.dirname(__file__))
        node_bin = _find_node_binary()
        print(f"Using Node.js binary: {node_bin}")
        return SubprocessMonitor(
            commands=[
                node_bin,
                U.f_join(file_path, "mineflayer/index.js"),
                str(server_port),
            ],
            name="mineflayer",
            ready_match=r"Server started on port (\d+)",
            log_path=U.f_join(self.log_path, "mineflayer"),
        )

    def get_mc_instance(self):
        print("Creating Minecraft server")
        U.f_mkdir(self.log_path, "minecraft")
        return MinecraftInstance(
            **self.azure_login,
            mineflayer=self.mineflayer,
            log_path=U.f_join(self.log_path, "minecraft"),
        )

    def check_process(self):
        if self.mc_instance and not self.mc_instance.is_running:
            # if self.mc_instance:
            #     self.mc_instance.check_process()
            #     if not self.mc_instance.is_running:
            print("Starting Minecraft server")
            self.mc_instance.run()
            self.mc_port = self.mc_instance.port
            self.reset_options["port"] = self.mc_instance.port
            print(f"Server started on port {self.reset_options['port']}")
        retry = 0
        while not self.mineflayer.is_running:
            print("Mineflayer process has exited, restarting")
            self.mineflayer.run()
            if not self.mineflayer.is_running:
                if retry > 3:
                    raise RuntimeError("Mineflayer process failed to start")
                else:
                    continue
            print(self.mineflayer.ready_line)
            res = requests.post(
                f"{self.server}/start",
                json=self.reset_options,
                timeout=self.request_timeout,
                headers=self._auth_headers,
            )
            if res.status_code != 200:
                self.mineflayer.stop()
                raise RuntimeError(
                    f"Minecraft server reply with code {res.status_code}"
                )
            return res.json()

    def step(
        self,
        code: str,
        programs: str = "",
    ) -> Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        if not self.has_reset:
            raise RuntimeError("Environment has not been reset yet")
        validate_code(code)
        self.check_process()
        if self.pause_between_steps:
            self.unpause()
        data = {
            "code": code,
            "programs": programs,
        }
        res = requests.post(
            f"{self.server}/step",
            json=data,
            timeout=self.request_timeout,
            headers=self._auth_headers,
        )
        if res.status_code != 200:
            raise RuntimeError("Failed to step Minecraft server")
        returned_data = res.json()
        if self.pause_between_steps:
            self.pause()
        return json.loads(returned_data)

    def render(self):
        raise NotImplementedError("render is not implemented")

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ) -> Tuple[ObsType, Dict[str, Any]]:
        if options is None:
            options = {}

        if options.get("inventory", {}) and options.get("mode", "hard") != "hard":
            raise RuntimeError("inventory can only be set when options is hard")

        self.reset_options = {
            "port": self.mc_port,
            "reset": options.get("mode", "hard"),
            "inventory": options.get("inventory", {}),
            "equipment": options.get("equipment", []),
            "spread": options.get("spread", False),
            "waitTicks": options.get("wait_ticks", 5),
            "position": options.get("position", None),
            "username": self.bot_username,
            "skin": self.bot_skin,
        }

        if self.pause_between_steps:
            self.unpause()
        self.mineflayer.stop()
        time.sleep(1)  # wait for mineflayer to exit

        returned_data = self.check_process()
        self.has_reset = True
        self.connected = True
        # All the reset in step will be soft
        self.reset_options["reset"] = "soft"
        if self.pause_between_steps:
            self.pause()
        return json.loads(returned_data)

    def close(self):
        if self.pause_between_steps:
            self.unpause()
        if self.connected:
            res = requests.post(
                f"{self.server}/stop", headers=self._auth_headers
            )
            if res.status_code == 200:
                self.connected = False
        if self.mc_instance:
            self.mc_instance.stop()
        self.mineflayer.stop()
        return not self.connected

    def pause(self):
        if self.mineflayer.is_running and not self.server_paused:
            res = requests.post(
                f"{self.server}/pause", headers=self._auth_headers
            )
            if res.status_code == 200:
                self.server_paused = True
        return self.server_paused

    def unpause(self):
        if self.mineflayer.is_running and self.server_paused:
            res = requests.post(
                f"{self.server}/unpause", headers=self._auth_headers
            )
            if res.status_code == 200:
                self.server_paused = False
            else:
                print(res.json())
        return self.server_paused
