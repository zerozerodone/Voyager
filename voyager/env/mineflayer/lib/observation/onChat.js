const Observation = require("./base.js").Observation;

class onChat extends Observation {
    constructor(bot) {
        super(bot);
        this.name = "onChat";
        this.obs = "";

        bot.on("chatEvent", (username, message) => {
            if (message.startsWith("/")) return;
            this.obs += `[${username}] ${message}\n`;
            this.bot.event(this.name);
        });

        bot.on("chat", (username, message) => {
            if (username === bot.username) return;
            if (message.startsWith("/")) return;
            this.obs += `[${username}] ${message}\n`;
            this.bot.event(this.name);
        });

        bot.on("whisper", (username, message) => {
            if (username === bot.username) return;
            this.obs += `[${username} whispers] ${message}\n`;
            this.bot.event(this.name);
        });
    }

    observe() {
        const result = this.obs.trim();
        this.obs = "";
        return result;
    }
}

module.exports = onChat;
