"""
Direct-controller agent for Minecraft.

The LLM picks from a fixed action menu each turn instead of generating
JavaScript.  Observations (text + screenshot) flow through unchanged;
the only thing that changes is how actions are selected and dispatched.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from voyager.prompts import load_prompt
from voyager.utils.json_utils import fix_and_parse_json

# ── Action catalogue ────────────────────────────────────────────────
# Organised by category.  Every Minecraft player verb that mineflayer
# can execute is represented here.

ACTIONS: dict[str, dict[str, Any]] = {

    # ── Resource gathering ───────────────────────────────────────
    "mine": {
        "description": "Mine/dig a block nearby. Bot walks to it, equips the best tool, and breaks it.",
        "params": {
            "block_name": "Block to mine (e.g. 'spruce_log', 'stone', 'iron_ore', 'diamond_ore')",
            "count": "How many to mine (default 1)",
        },
    },
    "collect": {
        "description": "Pick up a dropped item entity on the ground near the bot.",
        "params": {
            "item_name": "Item to collect (e.g. 'spruce_log', 'cobblestone')",
        },
    },

    # ── Crafting / smelting / enchanting ──────────────────────────
    "craft": {
        "description": "Craft an item. Uses a nearby crafting table if one exists.",
        "params": {
            "item_name": "Item to craft (e.g. 'spruce_planks', 'stick', 'wooden_pickaxe', 'chest')",
            "count": "How many crafting operations (default 1)",
        },
    },
    "smelt": {
        "description": "Smelt items in a nearby furnace.",
        "params": {
            "item_name": "Item to smelt (e.g. 'raw_iron', 'raw_gold', 'sand')",
            "fuel": "Fuel item (e.g. 'coal', 'spruce_planks', 'charcoal')",
            "count": "How many to smelt (default 1)",
        },
    },
    "enchant": {
        "description": "Enchant an item at a nearby enchantment table. Requires lapis lazuli and XP.",
        "params": {
            "choice": "Enchantment slot 0, 1, or 2 (0=cheapest, 2=most expensive)",
        },
    },
    "anvil_combine": {
        "description": "Combine two items at a nearby anvil (repair or merge enchantments).",
        "params": {
            "item_one": "First item name (the target)",
            "item_two": "Second item name (consumed)",
            "new_name": "Optional new name for the result (default: no rename)",
        },
    },

    # ── Block placement / interaction ────────────────────────────
    "place": {
        "description": "Place a block/item from inventory near the bot.",
        "params": {
            "item_name": "Item to place (e.g. 'crafting_table', 'furnace', 'chest', 'torch')",
        },
    },
    "activate_block": {
        "description": "Right-click / interact with a block (open door, press button, flip lever, ring bell, use note block).",
        "params": {
            "block_name": "Block type to find and activate",
        },
    },

    # ── Combat ───────────────────────────────────────────────────
    "kill": {
        "description": "Melee-attack and kill a nearby mob. Bot pathfinds to it and fights.",
        "params": {
            "mob_name": "Mob to fight (e.g. 'zombie', 'skeleton', 'spider', 'cow')",
            "timeout": "Max seconds to fight (default 300)",
        },
    },
    "shoot": {
        "description": "Ranged attack a mob with a bow, crossbow, trident, snowball, or egg.",
        "params": {
            "weapon": "Ranged weapon (e.g. 'bow', 'crossbow', 'trident', 'snowball')",
            "target": "Mob name to shoot (e.g. 'skeleton', 'creeper')",
        },
    },
    "shield": {
        "description": "Raise shield (use off-hand item) to block incoming damage. Deactivates after a duration.",
        "params": {
            "seconds": "How long to hold shield up (default 5)",
        },
    },

    # ── Equipment / inventory ────────────────────────────────────
    "equip": {
        "description": "Equip an inventory item to hand or armor slot.",
        "params": {
            "item_name": "Item to equip (e.g. 'wooden_pickaxe', 'iron_sword', 'iron_helmet')",
            "slot": "One of: hand, off-hand, head, torso, legs, feet (default hand)",
        },
    },
    "unequip": {
        "description": "Remove equipment from a slot.",
        "params": {
            "slot": "Slot to clear: hand, off-hand, head, torso, legs, feet",
        },
    },
    "drop": {
        "description": "Toss / discard items from inventory onto the ground.",
        "params": {
            "item_name": "Item to drop",
            "count": "How many to drop (default 1, use -1 for entire stack)",
        },
    },
    "swap_hands": {
        "description": "Swap the item between main hand and off-hand.",
        "params": {},
    },
    "select_hotbar": {
        "description": "Select a hotbar slot (0-8) as the active hand slot.",
        "params": {
            "slot": "Hotbar slot number 0-8",
        },
    },

    # ── Storage containers ───────────────────────────────────────
    "deposit": {
        "description": "Put items from inventory into the nearest chest.",
        "params": {
            "item_name": "Item to deposit",
            "count": "How many (default 1)",
        },
    },
    "withdraw": {
        "description": "Take items from the nearest chest into inventory.",
        "params": {
            "item_name": "Item to withdraw",
            "count": "How many (default 1)",
        },
    },

    # ── Movement / navigation ────────────────────────────────────
    "explore": {
        "description": "Walk in a direction, optionally searching for a specific block.",
        "params": {
            "direction": "north / south / east / west / up / down / random (default random)",
            "target_block": "Block to search for, or 'none' to just wander (default none)",
            "timeout": "Seconds to explore (default 60)",
        },
    },
    "go_to": {
        "description": "Navigate to specific x y z coordinates using pathfinding.",
        "params": {
            "x": "X coordinate",
            "y": "Y coordinate",
            "z": "Z coordinate",
        },
    },
    "go_near_entity": {
        "description": "Walk to within 3 blocks of a named entity (player or mob).",
        "params": {
            "entity_name": "Entity or player name to approach",
        },
    },
    "follow": {
        "description": "Continuously follow an entity or player at a set distance.",
        "params": {
            "entity_name": "Player or mob name to follow",
            "distance": "Blocks to stay away (default 3)",
            "timeout": "Seconds to follow before stopping (default 60)",
        },
    },
    "flee": {
        "description": "Run away from a named entity or mob.",
        "params": {
            "entity_name": "Entity to flee from",
            "distance": "How far to run (default 32)",
        },
    },
    "jump": {
        "description": "Jump once (or toggle continuous jumping).",
        "params": {},
    },
    "sprint": {
        "description": "Toggle sprint on or off.",
        "params": {
            "state": "true or false (default true)",
        },
    },
    "sneak": {
        "description": "Toggle sneaking on or off.",
        "params": {
            "state": "true or false (default true)",
        },
    },

    # ── Utility / world interaction ──────────────────────────────
    "eat": {
        "description": "Eat a food item from inventory.",
        "params": {"food_name": "Food to eat (e.g. 'cooked_beef', 'bread', 'golden_apple')"},
    },
    "fish": {
        "description": "Cast fishing rod and wait for a catch. Requires a fishing rod equipped.",
        "params": {},
    },
    "sleep": {
        "description": "Sleep in a nearby bed. Only works at night or during thunderstorms.",
        "params": {},
    },
    "wake": {
        "description": "Get out of bed.",
        "params": {},
    },
    "mount": {
        "description": "Mount a nearby vehicle, horse, boat, or minecart.",
        "params": {
            "entity_name": "Entity to mount (e.g. 'horse', 'boat', 'minecart')",
        },
    },
    "dismount": {
        "description": "Get off whatever vehicle/animal the bot is riding.",
        "params": {},
    },
    "use_item": {
        "description": "Generic right-click with held item (place boat, throw ender pearl, use bucket, etc.).",
        "params": {
            "off_hand": "true to use off-hand item instead of main hand (default false)",
        },
    },
    "look_at": {
        "description": "Turn to face specific coordinates.",
        "params": {
            "x": "X coordinate",
            "y": "Y coordinate",
            "z": "Z coordinate",
        },
    },

    # ── Social / multiplayer ─────────────────────────────────────
    "chat": {
        "description": "Send a chat message visible to all players.",
        "params": {"message": "Text to say"},
    },
    "whisper": {
        "description": "Send a private message to a specific player.",
        "params": {
            "player_name": "Player to whisper to",
            "message": "Text to send",
        },
    },
    "trade": {
        "description": "Trade with a nearby villager using one of their available trades.",
        "params": {
            "trade_index": "Trade slot number (0-based)",
            "times": "How many times to do the trade (default 1)",
        },
    },
    "give": {
        "description": "Toss an item toward a nearby player so they can pick it up.",
        "params": {
            "player_name": "Player to give item to",
            "item_name": "Item to toss",
            "count": "How many (default 1)",
        },
    },

    # ── Observation / passives ───────────────────────────────────
    "wait": {
        "description": "Do nothing for a short time. Useful for waiting/observing.",
        "params": {"seconds": "How long to wait (default 3)"},
    },
    "command": {
        "description": "Run a server slash-command (e.g. /time, /weather, /tp). Use sparingly.",
        "params": {"cmd": "The full command without the leading slash (e.g. 'time set day')"},
    },
}


# ── Helpers ──────────────────────────────────────────────────────────

_DIRECTION_MAP = {
    "north": (0, 0, -1),
    "south": (0, 0, 1),
    "east": (1, 0, 0),
    "west": (-1, 0, 0),
    "up": (0, 1, 0),
    "down": (0, -1, 0),
}


def _js(s: str) -> str:
    """Escape a value for embedding in a JS string literal."""
    return str(s).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _int(v, default: int = 1) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _float(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _bool(v, default: bool = True) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("true", "1", "yes") if v is not None else default


# ── Code templates ───────────────────────────────────────────────────

def action_to_code(name: str, params: dict) -> str:  # noqa: C901
    """Map an action name + params dict to a safe JavaScript snippet."""

    # ── Resource gathering ───────────────────────────────────────
    if name == "mine":
        block = _js(params.get("block_name", "dirt"))
        count = _int(params.get("count"), 1)
        return f'await mineBlock(bot, "{block}", {count});'

    if name == "collect":
        item = _js(params.get("item_name", ""))
        return (
            f'const _drop = bot.nearestEntity(e => e.name === "item" && '
            f'e.metadata?.[8]?.name === "{item}");\n'
            f'if (_drop) {{ await bot.pathfinder.goto(new GoalBlock('
            f'Math.floor(_drop.position.x), Math.floor(_drop.position.y), '
            f'Math.floor(_drop.position.z))); bot.chat("Collected {item}"); }}\n'
            f'else {{ bot.chat("No dropped {item} nearby"); }}'
        )

    # ── Crafting / smelting / enchanting ──────────────────────────
    if name == "craft":
        item = _js(params.get("item_name", ""))
        count = _int(params.get("count"), 1)
        return f'await craftItem(bot, "{item}", {count});'

    if name == "smelt":
        item = _js(params.get("item_name", ""))
        fuel = _js(params.get("fuel", "coal"))
        count = _int(params.get("count"), 1)
        return f'await smeltItem(bot, "{item}", "{fuel}", {count});'

    if name == "enchant":
        choice = _int(params.get("choice"), 0)
        return (
            f"const _eTable = bot.findBlock({{ matching: mcData.blocksByName.enchanting_table?.id, maxDistance: 32 }});\n"
            f"if (_eTable) {{\n"
            f"  const et = await bot.openEnchantmentTable(_eTable);\n"
            f"  const _held = bot.heldItem;\n"
            f"  if (_held) {{ await et.putTargetItem(_held); await et.putLapis(mcData.itemsByName.lapis_lazuli.id, null, 3);\n"
            f"    if (et.enchantments[{choice}]) {{ await et.enchant({choice}); bot.chat('Enchanted!'); }}\n"
            f"    else {{ bot.chat('Enchantment slot {choice} not available'); }}\n"
            f"    await et.takeTargetItem(); et.close(); }}\n"
            f"  else {{ bot.chat('Hold an item first'); et.close(); }}\n"
            f"}} else {{ bot.chat('No enchantment table nearby'); }}"
        )

    if name == "anvil_combine":
        i1 = _js(params.get("item_one", ""))
        i2 = _js(params.get("item_two", ""))
        new_name = params.get("new_name")
        name_arg = f', "{_js(new_name)}"' if new_name else ""
        return (
            f"const _anv = bot.findBlock({{ matching: mcData.blocksByName.anvil?.id, maxDistance: 32 }});\n"
            f"if (_anv) {{\n"
            f"  const av = await bot.openAnvil(_anv);\n"
            f'  const _i1 = bot.inventory.findInventoryItem(mcData.itemsByName["{i1}"]?.id);\n'
            f'  const _i2 = bot.inventory.findInventoryItem(mcData.itemsByName["{i2}"]?.id);\n'
            f"  if (_i1 && _i2) {{ await av.combine(_i1, _i2{name_arg}); bot.chat('Combined items'); }}\n"
            f"  else {{ bot.chat('Missing items for anvil'); }}\n"
            f"  av.close();\n"
            f"}} else {{ bot.chat('No anvil nearby'); }}"
        )

    # ── Block placement / interaction ────────────────────────────
    if name == "place":
        item = _js(params.get("item_name", ""))
        return (
            f"const pos = bot.entity.position.offset(1, 0, 0);\n"
            f'await placeItem(bot, "{item}", pos);'
        )

    if name == "activate_block":
        block = _js(params.get("block_name", ""))
        return (
            f'const _blk = bot.findBlock({{ matching: mcData.blocksByName["{block}"]?.id, maxDistance: 32 }});\n'
            f"if (_blk) {{ await bot.activateBlock(_blk); bot.chat('Activated {block}'); }}\n"
            f"else {{ bot.chat('No {block} nearby'); }}"
        )

    # ── Combat ───────────────────────────────────────────────────
    if name == "kill":
        mob = _js(params.get("mob_name", ""))
        timeout = _int(params.get("timeout"), 300)
        return f'await killMob(bot, "{mob}", {timeout});'

    if name == "shoot":
        weapon = _js(params.get("weapon", "bow"))
        target = _js(params.get("target", ""))
        return f'await shoot(bot, "{weapon}", "{target}");'

    if name == "shield":
        secs = _int(params.get("seconds"), 5)
        return (
            f"bot.activateItem(true);\n"
            f"await bot.waitForTicks({secs * 20});\n"
            f"bot.deactivateItem();\n"
            f'bot.chat("Shield lowered");'
        )

    # ── Equipment / inventory ────────────────────────────────────
    if name == "equip":
        item = _js(params.get("item_name", ""))
        slot = _js(params.get("slot", "hand"))
        return (
            f'const _item = bot.inventory.findInventoryItem(mcData.itemsByName["{item}"]?.id);\n'
            f'if (_item) {{ await bot.equip(_item, "{slot}"); bot.chat("Equipped {item}"); }}\n'
            f'else {{ bot.chat("No {item} in inventory"); }}'
        )

    if name == "unequip":
        slot = _js(params.get("slot", "hand"))
        return f'await bot.unequip("{slot}"); bot.chat("Unequipped {slot}");'

    if name == "drop":
        item = _js(params.get("item_name", ""))
        count = _int(params.get("count"), 1)
        if count < 0:
            return (
                f'const _d = bot.inventory.findInventoryItem(mcData.itemsByName["{item}"]?.id);\n'
                f"if (_d) {{ await bot.tossStack(_d); bot.chat('Dropped all {item}'); }}\n"
                f"else {{ bot.chat('No {item} to drop'); }}"
            )
        return (
            f'const _did = mcData.itemsByName["{item}"]?.id;\n'
            f"if (_did != null) {{ await bot.toss(_did, null, {count}); bot.chat('Dropped {count} {item}'); }}\n"
            f"else {{ bot.chat('Unknown item {item}'); }}"
        )

    if name == "swap_hands":
        return (
            "const _mh = bot.heldItem;\n"
            "const _oh = bot.inventory.slots[bot.getEquipmentDestSlot('off-hand')];\n"
            "if (_oh) { await bot.equip(_oh, 'hand'); }\n"
            "if (_mh) { await bot.equip(_mh, 'off-hand'); }\n"
            'bot.chat("Swapped hands");'
        )

    if name == "select_hotbar":
        slot = _int(params.get("slot"), 0)
        slot = max(0, min(8, slot))
        return f"bot.setQuickBarSlot({slot}); bot.chat('Selected hotbar {slot}');"

    # ── Storage containers ───────────────────────────────────────
    if name == "deposit":
        item = _js(params.get("item_name", ""))
        count = _int(params.get("count"), 1)
        return (
            f"const _chest = bot.findBlock({{ matching: mcData.blocksByName.chest?.id, maxDistance: 32 }});\n"
            f"if (_chest) {{\n"
            f'  await depositItemIntoChest(bot, _chest.position, {{"{item}": {count}}});\n'
            f"}} else {{ bot.chat('No chest nearby'); }}"
        )

    if name == "withdraw":
        item = _js(params.get("item_name", ""))
        count = _int(params.get("count"), 1)
        return (
            f"const _chest = bot.findBlock({{ matching: mcData.blocksByName.chest?.id, maxDistance: 32 }});\n"
            f"if (_chest) {{\n"
            f'  await getItemFromChest(bot, _chest.position, {{"{item}": {count}}});\n'
            f"}} else {{ bot.chat('No chest nearby'); }}"
        )

    # ── Movement / navigation ────────────────────────────────────
    if name == "explore":
        direction = str(params.get("direction", "random")).lower()
        target = str(params.get("target_block", "none")).lower()
        timeout = _int(params.get("timeout"), 60)
        if direction in _DIRECTION_MAP:
            dx, dy, dz = _DIRECTION_MAP[direction]
            dir_js = f"new Vec3({dx}, {dy}, {dz})"
        else:
            dir_js = (
                "([new Vec3(1,0,1),new Vec3(1,0,-1),new Vec3(-1,0,1),"
                "new Vec3(-1,0,-1)])[Math.floor(Math.random()*4)]"
            )
        if target and target != "none":
            safe_target = _js(target)
            callback = (
                f'() => bot.findBlock({{ matching: mcData.blocksByName["{safe_target}"]?.id, maxDistance: 32 }})'
            )
        else:
            callback = "() => null"
        return f"await exploreUntil(bot, {dir_js}, {timeout}, {callback});"

    if name == "go_to":
        x = int(_float(params.get("x")))
        y = int(_float(params.get("y")))
        z = int(_float(params.get("z")))
        return (
            f"bot.pathfinder.setMovements(new Movements(bot, mcData));\n"
            f"await bot.pathfinder.goto(new GoalBlock({x}, {y}, {z}));\n"
            f'bot.chat("Arrived at {x}, {y}, {z}");'
        )

    if name == "go_near_entity":
        ent = _js(params.get("entity_name", ""))
        return (
            f'const _ent = bot.nearestEntity(e => (e.name === "{ent}" || e.username === "{ent}"));\n'
            f"if (_ent) {{\n"
            f"  bot.pathfinder.setMovements(new Movements(bot, mcData));\n"
            f"  await bot.pathfinder.goto(new GoalNear(_ent.position.x, _ent.position.y, _ent.position.z, 3));\n"
            f'  bot.chat("Reached {ent}");\n'
            f"}} else {{ bot.chat('No {ent} nearby'); }}"
        )

    if name == "follow":
        ent = _js(params.get("entity_name", ""))
        dist = _int(params.get("distance"), 3)
        timeout = _int(params.get("timeout"), 60)
        return (
            f'const _fent = bot.nearestEntity(e => (e.name === "{ent}" || e.username === "{ent}"));\n'
            f"if (_fent) {{\n"
            f"  bot.pathfinder.setMovements(new Movements(bot, mcData));\n"
            f"  bot.pathfinder.setGoal(new GoalFollow(_fent, {dist}), true);\n"
            f"  await bot.waitForTicks({timeout * 20});\n"
            f"  bot.pathfinder.setGoal(null);\n"
            f'  bot.chat("Stopped following {ent}");\n'
            f"}} else {{ bot.chat('No {ent} nearby'); }}"
        )

    if name == "flee":
        ent = _js(params.get("entity_name", ""))
        dist = _int(params.get("distance"), 32)
        return (
            f'const _flee = bot.nearestEntity(e => (e.name === "{ent}" || e.username === "{ent}"));\n'
            f"if (_flee) {{\n"
            f"  bot.pathfinder.setMovements(new Movements(bot, mcData));\n"
            f"  await bot.pathfinder.goto(new GoalInvert(new GoalFollow(_flee, {dist})));\n"
            f'  bot.chat("Fled from {ent}");\n'
            f"}} else {{ bot.chat('No {ent} to flee from'); }}"
        )

    if name == "jump":
        return "bot.setControlState('jump', true); await bot.waitForTicks(4); bot.setControlState('jump', false);"

    if name == "sprint":
        state = "true" if _bool(params.get("state"), True) else "false"
        return f"bot.setControlState('sprint', {state});"

    if name == "sneak":
        state = "true" if _bool(params.get("state"), True) else "false"
        return f"bot.setControlState('sneak', {state});"

    # ── Utility / world interaction ──────────────────────────────
    if name == "eat":
        food = _js(params.get("food_name", ""))
        return (
            f'const _food = bot.inventory.findInventoryItem(mcData.itemsByName["{food}"]?.id);\n'
            f'if (_food) {{ await bot.equip(_food, "hand"); await bot.consume(); bot.chat("Ate {food}"); }}\n'
            f'else {{ bot.chat("No {food} in inventory"); }}'
        )

    if name == "fish":
        return (
            "const _rod = bot.inventory.findInventoryItem(mcData.itemsByName.fishing_rod?.id);\n"
            'if (_rod) { await bot.equip(_rod, "hand"); await bot.fish(); bot.chat("Caught something!"); }\n'
            'else { bot.chat("No fishing rod in inventory"); }'
        )

    if name == "sleep":
        return (
            "const _bed = bot.findBlock({ matching: b => mcData.blocks[b.type]?.name?.includes('bed'), maxDistance: 32 });\n"
            'if (_bed) { await bot.sleep(_bed); bot.chat("Sleeping..."); }\n'
            'else { bot.chat("No bed nearby"); }'
        )

    if name == "wake":
        return 'if (bot.isSleeping) { await bot.wake(); bot.chat("Woke up"); } else { bot.chat("Not sleeping"); }'

    if name == "mount":
        ent = _js(params.get("entity_name", ""))
        return (
            f'const _mv = bot.nearestEntity(e => e.name === "{ent}");\n'
            f'if (_mv) {{ bot.mount(_mv); bot.chat("Mounted {ent}"); }}\n'
            f'else {{ bot.chat("No {ent} nearby to mount"); }}'
        )

    if name == "dismount":
        return 'bot.dismount(); bot.chat("Dismounted");'

    if name == "use_item":
        off = "true" if _bool(params.get("off_hand"), False) else "false"
        return f"bot.activateItem({off}); await bot.waitForTicks(20); bot.deactivateItem();"

    if name == "look_at":
        x = _float(params.get("x"))
        y = _float(params.get("y"))
        z = _float(params.get("z"))
        return f"await bot.lookAt(new Vec3({x}, {y}, {z}), true);"

    # ── Social / multiplayer ─────────────────────────────────────
    if name == "chat":
        msg = _js(params.get("message", ""))
        return f'bot.chat("{msg}");'

    if name == "whisper":
        player = _js(params.get("player_name", ""))
        msg = _js(params.get("message", ""))
        return f'bot.whisper("{player}", "{msg}");'

    if name == "trade":
        idx = _int(params.get("trade_index"), 0)
        times = _int(params.get("times"), 1)
        return (
            f"const _vill = bot.nearestEntity(e => e.name === 'villager');\n"
            f"if (_vill) {{\n"
            f"  const v = await bot.openVillager(_vill);\n"
            f"  await bot.trade(v, {idx}, {times});\n"
            f"  v.close();\n"
            f"  bot.chat('Traded with villager');\n"
            f"}} else {{ bot.chat('No villager nearby'); }}"
        )

    if name == "give":
        player = _js(params.get("player_name", ""))
        item = _js(params.get("item_name", ""))
        count = _int(params.get("count"), 1)
        return (
            f'const _gp = bot.players["{player}"];\n'
            f"if (_gp?.entity) {{\n"
            f"  await bot.lookAt(_gp.entity.position);\n"
            f'  const _gid = mcData.itemsByName["{item}"]?.id;\n'
            f"  if (_gid != null) {{ await bot.toss(_gid, null, {count}); bot.chat('Gave {count} {item} to {player}'); }}\n"
            f"  else {{ bot.chat('Unknown item {item}'); }}\n"
            f"}} else {{ bot.chat('{player} not nearby'); }}"
        )

    # ── Observation / passives ───────────────────────────────────
    if name == "wait":
        secs = _int(params.get("seconds"), 3)
        return f"await bot.waitForTicks({secs * 20});"

    if name == "command":
        cmd = _js(params.get("cmd", ""))
        return f'bot.chat("/{cmd}");'

    return f'bot.chat("Unknown action: {_js(name)}");'


# ── Observation parser ───────────────────────────────────────────────

def parse_events(
    events: list, bot_username: str = "bot",
) -> tuple[str, list[str], list[str], list[str]]:
    """Return (observation_text, player_chat_lines, bot_chat_lines, error_lines).

    player_chat_lines are messages from other players (or whispers).
    bot_chat_lines are the bot's own feedback messages.
    """
    observe = None
    player_chats: list[str] = []
    bot_chats: list[str] = []
    errors: list[str] = []
    for event_type, event in events:
        if event_type == "observe":
            observe = event
        elif event_type == "onChat":
            raw = event.get("onChat", "")
            for line in raw.split("\n"):
                line = line.strip()
                if not line:
                    continue
                if line.startswith(f"[{bot_username}]"):
                    bot_chats.append(line)
                else:
                    player_chats.append(line)
        elif event_type == "onError":
            errors.append(event.get("onError", ""))

    if observe is None:
        return "No observation available.", player_chats, bot_chats, errors

    s = observe["status"]
    inv = observe["inventory"]
    voxels = observe["voxels"]
    entities = s.get("entities", {})
    pos = s["position"]

    lines = [
        f"Biome: {s['biome']}",
        f"Time: {s['timeOfDay']}",
        f"Health: {s['health']:.0f}/20",
        f"Hunger: {s['food']:.0f}/20",
        f"Position: x={pos['x']:.0f}, y={pos['y']:.0f}, z={pos['z']:.0f}",
        f"Equipment: {s['equipment']}",
    ]

    inv_used = s["inventoryUsed"]
    lines.append(f"Inventory ({inv_used}/36): {inv if inv else 'Empty'}")

    lines.append(f"Nearby blocks: {', '.join(voxels) if voxels else 'None'}")

    if entities:
        sorted_ents = sorted(entities.items(), key=lambda x: x[1])
        lines.append(f"Nearby entities: {', '.join(k for k, _ in sorted_ents)}")
    else:
        lines.append("Nearby entities: None")

    if player_chats:
        lines.append(f"Player chat:\n  " + "\n  ".join(player_chats))
    if bot_chats:
        lines.append(f"Bot feedback: {'; '.join(bot_chats)}")
    if errors:
        lines.append(f"Errors: {'; '.join(errors)}")

    return "\n".join(lines), player_chats, bot_chats, errors


# ── Agent ────────────────────────────────────────────────────────────

class PlayerAgent:
    def __init__(
        self,
        model_name: str = "gpt-4",
        temperature: float = 0.3,
        request_timeout: int = 240,
        max_memory: int = 10,
        enable_vision: bool = True,
        bot_username: str = "bot",
        thinking: bool = True,
    ):
        extra_kwargs: dict[str, Any] = {}
        if not thinking:
            extra_kwargs["model_kwargs"] = {"options": {"think": False}}

        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            request_timeout=request_timeout,
            **extra_kwargs,
        )
        self.memory: list[dict] = []
        self.max_memory = max_memory
        self.enable_vision = enable_vision
        self._vision_ok = enable_vision
        self.bot_username = bot_username
        self.pending_player_chats: list[str] = []

    # ── message building ─────────────────────────────────────────

    def _system_message(self) -> SystemMessage:
        base = load_prompt("player")
        action_docs = "\n\n## Available Actions\n\n"
        for name, info in ACTIONS.items():
            action_docs += f"**{name}** — {info['description']}\n"
            for pname, pdesc in info.get("params", {}).items():
                action_docs += f"  - {pname}: {pdesc}\n"
        return SystemMessage(content=base + action_docs)

    def _human_message(
        self, observation: str, screenshot: str | None, player_chats: list[str],
    ) -> HumanMessage:
        parts: list[str] = []

        if self.memory:
            history = []
            for i, t in enumerate(self.memory[-self.max_memory:], 1):
                entry = f"[Turn {i}] {t['action']}({json.dumps(t['params'], separators=(',', ':'))})"
                if t.get("result"):
                    entry += f" → {t['result'][:120]}"
                history.append(entry)
            parts.append("=== Recent Actions ===\n" + "\n".join(history))

        parts.append("=== Current State ===\n" + observation)

        if player_chats:
            parts.append(
                "=== INCOMING PLAYER MESSAGES (respond with chat action!) ===\n"
                + "\n".join(player_chats)
            )

        parts.append(
            'Pick your next action. Respond with JSON only: '
            '{"thought": "...", "action": "...", "params": {...}}'
        )
        content = "\n\n".join(parts)

        if screenshot and self._vision_ok:
            kb = len(screenshot) * 3 // 4 // 1024
            print(f"\033[36m[Vision] Player: screenshot ({kb} KB)\033[0m")
            return HumanMessage(content=[
                {"type": "text", "text": content},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{screenshot}"},
                },
            ])
        return HumanMessage(content=content)

    # ── core loop entry points ───────────────────────────────────

    def decide(self, events: list) -> str:
        """events from mineflayer → JavaScript code string to execute."""
        observation, player_chats, bot_chats, _ = parse_events(
            events, bot_username=self.bot_username,
        )
        all_player_chats = self.pending_player_chats + player_chats
        self.pending_player_chats = []

        if all_player_chats:
            print(f"\033[35m[Chat] Player messages: {all_player_chats}\033[0m")

        screenshot = self._extract_screenshot(events) if self.enable_vision else None

        messages = [
            self._system_message(),
            self._human_message(observation, screenshot, all_player_chats),
        ]

        try:
            response = self._invoke(messages)
            parsed = fix_and_parse_json(response.content)
        except Exception as e:
            print(f"\033[31m[Player] Parse error: {e}\033[0m")
            parsed = {"thought": "Error, observing", "action": "wait", "params": {"seconds": 3}}

        thought = parsed.get("thought", "")
        action_name = str(parsed.get("action", "wait"))
        params = parsed.get("params", {}) or {}

        if action_name not in ACTIONS:
            print(f"\033[31m[Player] Unknown action '{action_name}', waiting\033[0m")
            action_name, params = "wait", {"seconds": 3}

        code = action_to_code(action_name, params)

        print(f"\033[34m[Think] {thought}\033[0m")
        print(f"\033[32m[Act]   {action_name} {json.dumps(params, separators=(',', ':'))}\033[0m")

        self.memory.append({
            "observation_summary": observation[:200],
            "player_chats": all_player_chats,
            "thought": thought,
            "action": action_name,
            "params": params,
            "result": None,
        })
        if len(self.memory) > self.max_memory:
            self.memory = self.memory[-self.max_memory:]

        return code

    def record_result(self, events: list) -> None:
        """Call after env.step() to annotate the last memory entry with outcome."""
        if not self.memory:
            return
        _, player_chats, bot_chats, errors = parse_events(
            events, bot_username=self.bot_username,
        )
        if player_chats:
            self.pending_player_chats.extend(player_chats)
        parts = bot_chats + [f"ERR: {e}" for e in errors]
        self.memory[-1]["result"] = "; ".join(parts) if parts else "OK"

    # ── internal helpers ─────────────────────────────────────────

    @staticmethod
    def _extract_screenshot(events: list) -> str | None:
        for event_type, event in events:
            if event_type == "screenshot":
                return event
        return None

    def _invoke(self, messages):
        try:
            return self.llm.invoke(messages)
        except Exception as e:
            err = str(e).lower()
            if self._vision_ok and any(
                kw in err for kw in ("image", "vision", "multimodal", "content_type")
            ):
                print("\033[33m[Vision] Not supported by model, disabling.\033[0m")
                self._vision_ok = False
                stripped = []
                for msg in messages:
                    if isinstance(msg.content, list):
                        text = "\n".join(
                            p["text"] for p in msg.content if p.get("type") == "text"
                        )
                        stripped.append(msg.__class__(content=text))
                    else:
                        stripped.append(msg)
                return self.llm.invoke(stripped)
            raise
