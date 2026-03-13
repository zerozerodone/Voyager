const { Vec3 } = require("vec3");

let THREE, createCanvas, Viewer, WorldView, getBufferFromStream;
let depsAvailable = false;

try {
    THREE = require("three");
    createCanvas = require("node-canvas-webgl/lib").createCanvas;
    const viewerModule = require("prismarine-viewer").viewer;
    Viewer = viewerModule.Viewer;
    WorldView = viewerModule.WorldView;
    getBufferFromStream = viewerModule.getBufferFromStream;
    global.Worker = require("worker_threads").Worker;
    depsAvailable = true;
} catch (e) {
    console.warn(
        "[Screenshot] node-canvas-webgl or prismarine-viewer not available, screenshots disabled:",
        e.message
    );
}

const WIDTH = 512;
const HEIGHT = 512;
const VIEW_DISTANCE = 4;
const JPEG_QUALITY = 80;
const INIT_SETTLE_MS = 2000;

class Camera {
    constructor(bot) {
        this.bot = bot;
        this.ready = false;
        this.canvas = createCanvas(WIDTH, HEIGHT);
        this.renderer = new THREE.WebGLRenderer({ canvas: this.canvas });
        this.viewer = new Viewer(this.renderer);
        this.worldView = null;
    }

    async init() {
        const botPos = this.bot.entity.position;
        const center = new Vec3(botPos.x, botPos.y, botPos.z);
        this.viewer.setVersion(this.bot.version);

        this.worldView = new WorldView(
            this.bot.world,
            VIEW_DISTANCE,
            center
        );
        this.viewer.listen(this.worldView);
        this.viewer.camera.position.set(center.x, center.y, center.z);
        await this.worldView.init(center);

        await new Promise((resolve) => setTimeout(resolve, INIT_SETTLE_MS));
        this.ready = true;
    }

    async capture() {
        if (!this.ready) return null;

        const pos = this.bot.entity.position;
        const yaw = this.bot.entity.yaw;
        const pitch = this.bot.entity.pitch;

        this.viewer.camera.position.set(pos.x, pos.y + 1.62, pos.z);

        const lookX = pos.x - Math.sin(yaw) * Math.cos(pitch);
        const lookY = pos.y + 1.62 - Math.sin(pitch);
        const lookZ = pos.z + Math.cos(yaw) * Math.cos(pitch);
        this.viewer.camera.lookAt(lookX, lookY, lookZ);

        if (this.worldView) {
            await this.worldView.updatePosition(pos);
        }

        this.renderer.render(this.viewer.scene, this.viewer.camera);

        const stream = this.canvas.createJPEGStream({
            bufsize: 4096,
            quality: JPEG_QUALITY,
            progressive: false,
        });
        const buf = await getBufferFromStream(stream);
        return buf.toString("base64");
    }
}

function inject(bot) {
    if (!depsAvailable) {
        bot._camera = null;
        bot.screenshot = async function () {
            return null;
        };
        return;
    }

    bot._camera = null;

    bot.screenshot = async function () {
        if (!bot._camera) {
            try {
                bot._camera = new Camera(bot);
                await bot._camera.init();
            } catch (e) {
                console.warn("[Screenshot] Failed to initialize camera:", e.message);
                bot._camera = null;
                return null;
            }
        }
        try {
            return await bot._camera.capture();
        } catch (e) {
            console.warn("[Screenshot] Failed to capture:", e.message);
            return null;
        }
    };
}

module.exports = { inject };
