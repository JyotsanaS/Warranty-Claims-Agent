# AI Usage

## Tools

- Excalidraw
- Claude Code
- Codex

## Notes

### Planning / architecture

- Drew initial HLD in Excalidraw first
- Used that diagram with Claude Code to draft `ARCHITECTURE.md`
- Claude Code helped expand components / flow quickly
- Claude Code missed some important product/state details
- I came up with the planner-agent idea and execution flow myself.
- I decided the final conversation-close behavior and end-of-flow handling myself.

Missing pieces I had to add or think through:
- how to track different claims made by a user
- what claim info needs to be captured over the conversation
- what evidence is mandatory before decisioning
- how text, image analysis, and policy retrieval connect into one claim flow
- I worked on the custom chunking logic myself.
- Retrieval did not have any major AI contribution.
