"""
Direct-controller play mode for Voyager.

The bot connects to Minecraft and plays continuously — no task curriculum,
no code generation, no game pausing.  The LLM observes (text + screenshots)
and picks actions from a fixed menu each turn.

Usage:
    uv run voyager_play.py
"""

import os
import time
import warnings
import urllib3

warnings.filterwarnings("ignore", category=urllib3.exceptions.NotOpenSSLWarning)
warnings.filterwarnings("ignore", message="urllib3.*doesn't match a supported version")

from voyager.env import VoyagerEnv
from voyager.agents.player import PlayerAgent
from voyager.control_primitives import load_control_primitives

# ── Configuration ────────────────────────────────────────────────────

os.environ["OPENAI_API_BASE"] = "http://localhost:11434/v1"
os.environ["OPENAI_API_KEY"] = "ollama"

MODEL = "magistral:24b-small-2509"
MC_PORT = 25565
THINKING = False

BOT_USERNAME = "Voyager"
BOT_SKIN = "Technoblade"

# ── Setup ────────────────────────────────────────────────────────────

programs = "\n\n".join(load_control_primitives())

env = VoyagerEnv(
    mc_port=MC_PORT,
    pause_between_steps=False,
    bot_username=BOT_USERNAME,
    bot_skin=BOT_SKIN,
)

agent = PlayerAgent(
    model_name=MODEL,
    temperature=0.3,
    max_memory=10,
    bot_username=BOT_USERNAME,
    thinking=THINKING,
)

# ── Connect and start ───────────────────────────────────────────────

print(f"Connecting to Minecraft as '{BOT_USERNAME}' (skin: {BOT_SKIN})...")
events = env.reset(options={"mode": "soft", "wait_ticks": 20})

events = env.step(
    'bot.chat("/gamerule doDaylightCycle true");\n'
    'bot.chat("/difficulty easy");',
    programs=programs,
)

print("Bot connected. Playing — press Ctrl+C to stop.\n")

turn = 0
while True:
    turn += 1
    try:
        t0 = time.time()
        code = agent.decide(events)
        think_time = time.time() - t0

        t1 = time.time()
        events = env.step(code, programs=programs)
        act_time = time.time() - t1

        agent.record_result(events)

        print(
            f"\033[90m  turn {turn}  think {think_time:.1f}s  act {act_time:.1f}s\033[0m\n"
        )

    except KeyboardInterrupt:
        print("\nStopping...")
        env.close()
        break

    except Exception as e:
        print(f"\033[41mError: {e}\033[0m")
        try:
            time.sleep(3)
            events = env.reset(options={"mode": "soft", "wait_ticks": 20})
            events = env.step("", programs=programs)
        except Exception:
            print("Could not recover, exiting.")
            env.close()
            break
