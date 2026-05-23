const { Vec3 } = require("vec3");

let createCanvas;
let canvasAvailable = false;
let puppeteer;
let prismarineViewer;

try {
    createCanvas = require("canvas").createCanvas;
    canvasAvailable = true;
} catch (e) {
    console.warn("[Screenshot] canvas not available, map disabled:", e.message);
}

try {
    puppeteer = require("puppeteer-core");
    prismarineViewer = require("prismarine-viewer");
} catch (e) {
    console.warn("[Screenshot] puppeteer-core or prismarine-viewer not available, 3D view disabled:", e.message);
}

// ── Constants ──────────────────────────────────────────────────────

const MAP_WIDTH = 512;
const MAP_HEIGHT = 512;
const MAP_RADIUS = 32;
const FPV_WIDTH = 800;
const FPV_HEIGHT = 600;
const JPEG_QUALITY = 80;
const VIEWER_PORT = 3007;

const CHROME_PATHS = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium-browser",
    "/usr/bin/chromium",
];

// ── Block color map for 2D overhead view ───────────────────────────

const BLOCK_COLORS = {
    air: null, cave_air: null, void_air: null,

    grass_block: "#5d9b47", short_grass: "#68a84a", tall_grass: "#68a84a",
    dirt: "#8b6b4a", coarse_dirt: "#77593b", farmland: "#6b4f31",
    dirt_path: "#937545", mycelium: "#6b6162", podzol: "#5a3d1e", mud: "#3c3537",

    stone: "#7f7f7f", cobblestone: "#7a7a7a", mossy_cobblestone: "#6a7a5a",
    deepslate: "#505050", cobbled_deepslate: "#4a4a4a",
    granite: "#9a6c50", diorite: "#bfbfbf", andesite: "#888888",
    tuff: "#6b6b5f", calcite: "#ddddd0", dripstone_block: "#7a6855",
    bedrock: "#333333", gravel: "#858080", clay: "#9ea4b0",

    sand: "#dbd3a0", sandstone: "#d4c484", red_sand: "#b5602a",
    red_sandstone: "#a04020", soul_sand: "#4b3a29", soul_soil: "#4b3a29",

    oak_log: "#6b5230", spruce_log: "#3b2a14", birch_log: "#d5cca3",
    jungle_log: "#56440f", acacia_log: "#676058", dark_oak_log: "#3b2a0e",
    mangrove_log: "#5a3024", cherry_log: "#2d1215",
    oak_planks: "#a08050", spruce_planks: "#6b5030", birch_planks: "#c4b77a",
    jungle_planks: "#a08050", acacia_planks: "#a05830", dark_oak_planks: "#3b2a14",
    mangrove_planks: "#773530", cherry_planks: "#e0b0a0",

    oak_leaves: "#3a6a20", spruce_leaves: "#2a5a30", birch_leaves: "#5a8a30",
    jungle_leaves: "#2a7a10", acacia_leaves: "#3a7a20", dark_oak_leaves: "#2a5a10",
    azalea_leaves: "#5a7a30", mangrove_leaves: "#3a7a20", cherry_leaves: "#e8a8c0",

    water: "#3366cc", lava: "#cf5a00",
    ice: "#8ab4e8", packed_ice: "#7aa4e0", blue_ice: "#6a94e0",
    snow: "#f0f0f0", snow_block: "#f0f0f0", powder_snow: "#f8f8f8",

    coal_ore: "#303030", iron_ore: "#b09080", gold_ore: "#fcee4b",
    diamond_ore: "#5decf5", emerald_ore: "#41f384", lapis_ore: "#2060c0",
    redstone_ore: "#aa0000", copper_ore: "#c07050",
    deepslate_coal_ore: "#252525", deepslate_iron_ore: "#907060",
    deepslate_gold_ore: "#d0c040", deepslate_diamond_ore: "#50d0e0",
    deepslate_emerald_ore: "#30d070", deepslate_lapis_ore: "#1a50a0",
    deepslate_redstone_ore: "#900000", deepslate_copper_ore: "#a06040",
    ancient_debris: "#5a3d27", nether_gold_ore: "#d4a800",

    crafting_table: "#b08040", furnace: "#6a6a6a", chest: "#9a7a40",
    ender_chest: "#1a2a3a", barrel: "#8a6a3a", anvil: "#444444",
    enchanting_table: "#3a1a6a", brewing_stand: "#5a5a3a",
    torch: "#ffc800", lantern: "#e8a020", campfire: "#ff6600",

    obsidian: "#1a0a2a", crying_obsidian: "#3a0a5a", netherrack: "#702020",
    basalt: "#484848", blackstone: "#2a2024", end_stone: "#d8d8a0",
    glowstone: "#ffcc66", nether_bricks: "#301818", prismarine: "#5a9a8a",

    cactus: "#0a6a0a", sugar_cane: "#7acc6b", bamboo: "#5a8a30",
    melon: "#6da020", pumpkin: "#d08020", hay_block: "#b0a020", wheat: "#a0a020",
};

const ENTITY_COLORS = {
    zombie: "#00aa00", skeleton: "#cccccc", creeper: "#00cc00",
    spider: "#443333", enderman: "#220022", witch: "#550088",
    phantom: "#4466aa", blaze: "#ff8800", ghast: "#eeeeee",
    wither_skeleton: "#333333", piglin: "#c8a050",
    cow: "#6b4226", pig: "#e8a0a0", sheep: "#e0ddd0",
    chicken: "#ffffff", horse: "#8b6914", wolf: "#b0b0b0",
    cat: "#c08040", rabbit: "#a08060", villager: "#8b6914",
    iron_golem: "#c0c0c0", player: "#ff4444", item: "#ffff00",
};

function hexToRgb(hex) {
    return [
        parseInt(hex.slice(1, 3), 16),
        parseInt(hex.slice(3, 5), 16),
        parseInt(hex.slice(5, 7), 16),
    ];
}

// ── 2D Map Renderer (canvas, no GPU) ───────────────────────────────

class MapRenderer {
    constructor(bot) {
        this.bot = bot;
        this.canvas = createCanvas(MAP_WIDTH, MAP_HEIGHT);
        this.ctx = this.canvas.getContext("2d");
    }

    capture() {
        const ctx = this.ctx;
        const bot = this.bot;
        const pos = bot.entity.position;
        const yaw = bot.entity.yaw;
        const cx = Math.floor(pos.x);
        const cy = Math.floor(pos.y);
        const cz = Math.floor(pos.z);
        const scale = MAP_WIDTH / (MAP_RADIUS * 2);

        ctx.fillStyle = "#1a1a2e";
        ctx.fillRect(0, 0, MAP_WIDTH, MAP_HEIGHT);

        for (let dx = -MAP_RADIUS; dx < MAP_RADIUS; dx++) {
            for (let dz = -MAP_RADIUS; dz < MAP_RADIUS; dz++) {
                let color = null;
                let surfaceY = cy;
                for (let dy = 10; dy >= -10; dy--) {
                    const block = bot.blockAt(new Vec3(cx + dx, cy + dy, cz + dz));
                    if (block && block.name !== "air" && block.name !== "cave_air") {
                        surfaceY = cy + dy;
                        color = BLOCK_COLORS[block.name] !== undefined
                            ? BLOCK_COLORS[block.name] : "#888888";
                        break;
                    }
                }
                if (!color) continue;

                const [r, g, b] = hexToRgb(color);
                const shade = Math.max(0.4, Math.min(1.2, 1.0 + (surfaceY - cy) * 0.03));
                ctx.fillStyle = `rgb(${Math.min(255, Math.floor(r * shade))},${Math.min(255, Math.floor(g * shade))},${Math.min(255, Math.floor(b * shade))})`;
                ctx.fillRect((dx + MAP_RADIUS) * scale, (dz + MAP_RADIUS) * scale, scale + 0.5, scale + 0.5);
            }
        }

        const entities = Object.values(bot.entities);
        for (const entity of entities) {
            if (entity === bot.entity) continue;
            const ex = entity.position.x - cx;
            const ez = entity.position.z - cz;
            if (Math.abs(ex) > MAP_RADIUS || Math.abs(ez) > MAP_RADIUS) continue;
            const px = (ex + MAP_RADIUS) * scale;
            const py = (ez + MAP_RADIUS) * scale;
            const name = entity.name || entity.username || "unknown";
            ctx.fillStyle = ENTITY_COLORS[name] || (entity.type === "hostile" ? "#ff0000" : "#ffaa00");
            const sz = entity.type === "player" ? 6 : 4;
            ctx.fillRect(px - sz / 2, py - sz / 2, sz, sz);
            ctx.fillStyle = "#ffffff";
            ctx.font = "bold 10px sans-serif";
            ctx.textAlign = "center";
            ctx.fillText(entity.username || name, px, py - sz);
        }

        const bpx = MAP_RADIUS * scale;
        const bpy = MAP_RADIUS * scale;
        ctx.save();
        ctx.translate(bpx, bpy);
        ctx.rotate(-yaw + Math.PI);
        ctx.fillStyle = "#ff0000";
        ctx.beginPath();
        ctx.moveTo(0, -8);
        ctx.lineTo(-5, 6);
        ctx.lineTo(5, 6);
        ctx.closePath();
        ctx.fill();
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 1.5;
        ctx.stroke();
        ctx.restore();

        ctx.fillStyle = "rgba(0, 0, 0, 0.6)";
        ctx.fillRect(0, 0, MAP_WIDTH, 20);
        ctx.fillStyle = "#ffffff";
        ctx.font = "12px sans-serif";
        ctx.textAlign = "left";
        ctx.fillText(
            `x=${Math.floor(pos.x)} y=${Math.floor(pos.y)} z=${Math.floor(pos.z)}  HP:${Math.floor(bot.health)}/20  Food:${Math.floor(bot.food)}/20`,
            5, 14
        );

        ctx.fillStyle = "#aaaaaa";
        ctx.font = "10px sans-serif";
        ctx.textAlign = "right";
        ctx.fillText("N", MAP_WIDTH / 2, MAP_HEIGHT - 4);
        ctx.fillText("S", MAP_WIDTH / 2, 32);
        ctx.fillText("W", MAP_WIDTH - 4, MAP_HEIGHT / 2);
        ctx.fillText("E", 12, MAP_HEIGHT / 2);

        return this.canvas.toBuffer("image/jpeg", { quality: JPEG_QUALITY / 100 }).toString("base64");
    }
}

// ── 3D First-Person Renderer (prismarine-viewer + Puppeteer) ───────

class FPVRenderer {
    constructor(bot) {
        this.bot = bot;
        this.browser = null;
        this.page = null;
        this.viewerServer = null;
        this.ready = false;
        this.initPromise = null;
    }

    async init() {
        const fs = require("fs");
        const chromePath = CHROME_PATHS.find((p) => {
            try { return fs.existsSync(p); } catch { return false; }
        });
        if (!chromePath) {
            throw new Error("No Chrome/Chromium binary found");
        }

        // prismarine-viewer.mineflayer() creates bot.viewer and starts an
        // HTTP server.  It binds on 0.0.0.0 and the http 'error' event is
        // unhandled inside the library, so we must patch it.
        const http = require("http");
        const origListen = http.Server.prototype.listen;
        let viewerHttpServer = null;
        http.Server.prototype.listen = function (...args) {
            viewerHttpServer = this;
            return origListen.apply(this, args);
        };

        prismarineViewer.mineflayer(this.bot, {
            viewDistance: 6,
            firstPerson: true,
            port: VIEWER_PORT,
        });

        http.Server.prototype.listen = origListen;

        // Wait for the server to start, handling bind errors gracefully
        await new Promise((resolve, reject) => {
            if (!viewerHttpServer) return reject(new Error("Viewer did not create HTTP server"));
            viewerHttpServer.on("error", (err) => reject(err));
            viewerHttpServer.on("listening", () => resolve());
            setTimeout(() => resolve(), 3000);
        });

        this.browser = await puppeteer.launch({
            executablePath: chromePath,
            headless: "new",
            args: [
                `--window-size=${FPV_WIDTH},${FPV_HEIGHT}`,
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
                "--disable-extensions",
                "--mute-audio",
            ],
        });

        this.page = await this.browser.newPage();
        await this.page.setViewport({ width: FPV_WIDTH, height: FPV_HEIGHT });
        await this.page.goto(`http://127.0.0.1:${VIEWER_PORT}`, {
            waitUntil: "networkidle0",
            timeout: 15000,
        });

        await new Promise((resolve) => setTimeout(resolve, 3000));
        this.ready = true;
        console.log("[Screenshot] 3D first-person view initialized");
    }

    async capture() {
        if (!this.ready) return null;
        try {
            const buf = await this.page.screenshot({
                type: "jpeg",
                quality: JPEG_QUALITY,
                encoding: "binary",
            });
            return buf.toString("base64");
        } catch (e) {
            console.warn("[Screenshot] 3D capture failed:", e.message);
            return null;
        }
    }

    async close() {
        if (this.browser) {
            try { await this.browser.close(); } catch {}
            this.browser = null;
        }
        if (this.viewerServer) {
            try { this.viewerServer.close(); } catch {}
            this.viewerServer = null;
        }
        this.ready = false;
    }
}

// ── Injection ──────────────────────────────────────────────────────

function inject(bot) {
    let mapRenderer = null;
    let fpvRenderer = null;
    let fpvFailed = false;

    const canDo3D = !!(puppeteer && prismarineViewer);

    bot.screenshot = async function () {
        const results = {};

        // 2D map (synchronous, fast)
        if (canvasAvailable) {
            try {
                if (!mapRenderer) mapRenderer = new MapRenderer(bot);
                results.map = mapRenderer.capture();
            } catch (e) {
                console.warn("[Screenshot] Map capture failed:", e.message);
            }
        }

        // 3D first-person (async, uses browser)
        if (canDo3D && !fpvFailed) {
            try {
                if (!fpvRenderer) {
                    fpvRenderer = new FPVRenderer(bot);
                    await fpvRenderer.init();
                }
                results.fpv = await fpvRenderer.capture();
            } catch (e) {
                console.warn("[Screenshot] 3D init/capture failed:", e.message);
                fpvFailed = true;
                if (fpvRenderer) {
                    await fpvRenderer.close();
                    fpvRenderer = null;
                }
            }
        }

        if (results.fpv && results.map) {
            return { fpv: results.fpv, map: results.map };
        }
        if (results.fpv) return results.fpv;
        if (results.map) return results.map;
        return null;
    };

    const origEnd = bot.end.bind(bot);
    bot.end = function (...args) {
        if (fpvRenderer) {
            fpvRenderer.close().catch(() => {});
            fpvRenderer = null;
        }
        return origEnd(...args);
    };
}

module.exports = { inject };
