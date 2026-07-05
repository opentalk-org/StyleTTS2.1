i want dynamic generation of workflows jsons, give me more exact plan. runner and runflow shouldn't know of existence of "actions". So for example:
I can select few audios manually / choose all audios from given dataset / choose all audio node -> load audio node from bucket -> then it get to for example whisper node -> then it get's to save transcriptions node.

frontend just aggregate settings of these nodes and render ui based on that (reuse the same components as workflow view)

The runner/runflow shouldn't know of existence of actions.

I want you to start by porting workflow ui to current react, divide things in reusable things. also change left navbar with components to be single button at botto (bottom navbar) that open popup that allow for choosing node. the global settings work by second button on bottom navbar (popup).

Create detailed plan. The backend should support multiple runners. modify nix so it runs as many runners as there are gpus and auto connect them to backend. (there is tab for that "cluster" (change it from ray to just runners) also allow adding extra.). Also add s3 bucket settings (set to rustfs by default) to settings.

After every step of plan, read this design again and check what is not implemented.