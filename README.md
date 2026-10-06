# OptChat

OptChat is a Pi extension for one ongoing chat with persistent memory. Every message is kept in an append-only log, while a background compactor builds a binary tree of summaries. Each turn starts fresh with a bounded view of the conversation, and the agent can zoom in on older details when needed.

Minimal implementation of https://gist.github.com/VictorTaelin/91837951a5ce5b38f341ec1ba1df6449 for a trial run of the strategy. Omits subagent orchestration and advanced features.

All credit to Victor on this one.

## Why

Typical chat compaction discards detail, while separate memory files require manual curation and can lose context. OptChat keeps the original history and summarizes it at multiple levels, so the agent can recover specifics without loading the whole conversation every turn.

## Usage

```sh
pi --extension ./extensions/optchat.ts \
  --optchat-identity MyAgent \
  --optchat-dir ./optchat-memory \
  [--optchat-model <model_name>]
```

Import existing notes with `/optchat-import <file-path> [--topic <topic>]`. Use `/optchat-info` to check memory status and `/optchat-export` to browse the full history.

## License

MIT cuz who cares.

Written by Gemini 3.6 Thinking + GPT 6 Luna. Not my handiwork.
