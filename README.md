# OptChat Pi Extension

Implementation of https://gist.github.com/VictorTaelin/91837951a5ce5b38f341ec1ba1df6449 for a trial run of the strategy.

All credit to Victor on this one.

## tl;dr

```sh
# Run the extension under a specific identity and memory storage.
pi --extension ./extensions/optchat.ts \
  --optchat-identity Chasebot \
  --optchat-dir /workspace/chasebot_memory \
  [--optchat-model <model_name>]
```

This extension targets the pi 1 SDK.

Use `/optchat-import <file-path> [--topic <topic>]` to import JSON, JSONL,
Markdown, or plain-text notes. Quote paths or topics that contain spaces. The
agent can also save durable memories with the `write_note` tool.

Written by Gemini 3.6 Thinking + GPT 6 Luna. Not my handiwork.

## License

MIT cuz who cares.
