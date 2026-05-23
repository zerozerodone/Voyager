"""
End-to-end vision smoke test against a live Minecraft server.

Connects the bot, captures screenshots (map + 3D FPV), verifies both
appear in the [Vision] diagnostic, and runs one LLM turn.

Usage:  uv run smoke_test_vision.py
"""

import os
import sys
import time
import warnings
import urllib3

warnings.filterwarnings("ignore", category=urllib3.exceptions.NotOpenSSLWarning)
warnings.filterwarnings("ignore", message="urllib3.*doesn't match a supported version")

os.environ["OPENAI_API_BASE"] = "http://localhost:11434/v1"
os.environ["OPENAI_API_KEY"] = "ollama"

from voyager.env import VoyagerEnv
from voyager.agents.player import PlayerAgent
from voyager.control_primitives import load_control_primitives

MC_PORT = 25565
MODEL = "gemma4:26b-arush"
BOT_USERNAME = "Voyager"

passed = 0
failed = 0


def check(label, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  \033[32mPASS\033[0m {label}" + (f" — {detail}" if detail else ""))
    else:
        failed += 1
        print(f"  \033[31mFAIL\033[0m {label}" + (f" — {detail}" if detail else ""))


print("=" * 60)
print("Vision Smoke Test — live server on port", MC_PORT)
print("=" * 60)

# ── 1. Connect to MC ───────────────────────────────────────────────

print("\n[1/5] Connecting bot to Minecraft...")
programs = "\n\n".join(load_control_primitives())
env = VoyagerEnv(
    mc_port=MC_PORT,
    pause_between_steps=False,
    bot_username=BOT_USERNAME,
)

try:
    events = env.reset(options={"mode": "soft", "wait_ticks": 20})
    check("Bot connected", True)
except Exception as e:
    check("Bot connected", False, str(e))
    env.close()
    sys.exit(1)

# ── 2. Run an initial step to generate events with screenshot ──────

print("\n[2/5] Running initial step to capture screenshot...")
try:
    events = env.step(
        'bot.chat("Vision test");',
        programs=programs,
    )
    check("Step executed", True)
except Exception as e:
    check("Step executed", False, str(e))
    env.close()
    sys.exit(1)

# ── 3. Inspect raw screenshot data from events ────────────────────

print("\n[3/5] Inspecting screenshot data from events...")
screenshot_data = None
for event_type, event in events:
    if event_type == "screenshot":
        screenshot_data = event
        break

check("Screenshot event present", screenshot_data is not None)

if isinstance(screenshot_data, dict):
    fpv = screenshot_data.get("fpv")
    map_img = screenshot_data.get("map")
    check("FPV image captured", fpv is not None and len(fpv) > 100,
          f"{len(fpv) * 3 // 4 // 1024} KB" if fpv else "missing")
    check("Map image captured", map_img is not None and len(map_img) > 100,
          f"{len(map_img) * 3 // 4 // 1024} KB" if map_img else "missing")
elif isinstance(screenshot_data, str) and len(screenshot_data) > 100:
    check("Single image captured (map only, 3D unavailable)", True,
          f"{len(screenshot_data) * 3 // 4 // 1024} KB")
    check("FPV image captured", False, "3D renderer did not start — check Chrome/port")
else:
    check("Screenshot data valid", False, f"type={type(screenshot_data)}")

# ── 4. Feed through PlayerAgent ────────────────────────────────────

print("\n[4/5] Running PlayerAgent.decide() with live screenshot...")
agent = PlayerAgent(
    model_name=MODEL,
    temperature=0.3,
    max_memory=10,
    bot_username=BOT_USERNAME,
    thinking=False,
)

t0 = time.time()
try:
    code = agent.decide(events)
    elapsed = time.time() - t0
    check("LLM responded", bool(code), f"{elapsed:.1f}s")
    check("Vision still enabled", agent._vision_ok)
    print(f"\n  Generated JS: {code[:200]}{'...' if len(code) > 200 else ''}")
except Exception as e:
    check("LLM responded", False, str(e))

# ── 5. Execute the generated code ─────────────────────────────────

print("\n[5/5] Executing generated action in Minecraft...")
try:
    result_events = env.step(code, programs=programs)
    agent.record_result(result_events)
    last_result = agent.memory[-1]["result"] if agent.memory else "no memory"
    check("Action executed", True, last_result)
except Exception as e:
    check("Action executed", False, str(e))

# ── Cleanup ────────────────────────────────────────────────────────

print("\nCleaning up...")
env.close()

print("\n" + "=" * 60)
print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
if failed:
    print("\033[31mSome tests failed.\033[0m")
    sys.exit(1)
else:
    print("\033[32mAll tests passed!\033[0m")
