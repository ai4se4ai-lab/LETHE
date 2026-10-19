# Prompt policy

Prompts describe the task and tool protocol without prescribing a localization strategy. First turns
receive the full task; later turns receive status only. A post-forget turn is a new first turn. Prompt
contents and hashes are recorded in the run manifest. Ground truth, fixed code, gold patches, and VCS
history are forbidden from every agent-visible workspace.
