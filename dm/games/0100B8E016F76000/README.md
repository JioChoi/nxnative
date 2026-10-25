# NieR:Automata The End of YoRHa Edition (`0100B8E016F76000`)

Base game v0 (display version 1.0.1), SDK 12.3.9. PlatinumGames' engine (the Switch port is Virtuos') on NVN. Modules `main`, `subsdk0` (`NintendoSDK_movie`: the logo movies, VP9 in CRI USM through `movie::Decoder`) and `sdk`. CRI's file system (CPK), Wwise sound, Havok, Enlighten. **Work in progress**: boots and plays, but models render black on the iPhone (GitHub issue #163).

On the Mac (fresh save): the boot, the logo movies, the "booting system" screen, the title, a new game (slot 1, the auto-save notice, a name through swkbd, Easy or Normal), the opening text and the prologue's flight-unit section over the sea and in the clouds (shooting, explosions, enemy bullets, HUD, the damage glitch effect), up to a game over. 30 fps as on the Switch. About 2.95 GB of the 3.21 GB pool is mapped at boot and stays there.

## What it needs

| Setting | Where | Why |
|---|---|---|
| 5 GB block: `mem_reserve 0x140000000`, `mem_alias 0x118000000`, `mem_stack 0x10000000`, `app_pool 0xCD500000` | [`game.tsv`](game.tsv) | the NPDM declares 16 MB of system resource and `main` sets no heap: `nninitStartup` gives StandardAllocator a 1 MB static block and the engine maps its memory in the alias region (up to +4.05 GB within seconds, then flat; no region request fails, so no node-area cut) |
| `sys_resource 0x1000000` | `game.tsv` | the NPDM's system resource |
| `tex_flush_upload on`, `tex_dirty_upload on` | `game.tsv` | the title background (ASTC 6x6 sRGB) is in a `CPU_UNCACHED \| GPU_CACHED` pool, filled by the CPU after `nvnTextureInitialize` with no flush; without both keys it read as all-zero ASTC blocks: a magenta title screen |
| `feedback_snap fd2c47b73f0e7e24` | `game.tsv` | FXAA samples the scene target it renders into ([Same-pass feedback read through a snapshot](../../../docs/reference/rendering.md#same-pass-feedback-read-through-a-snapshot)) |
| `pad_layout full` | `game.tsv` | the camera is on the right stick |
| `shader_groups` by DAT | `game.tsv` | `shader.dat`, `shader2.dat`, the two effect DATs and `autoshadereff.dat` |

## Files and shaders

- 30 CRI CPK archives in the RomFS root (`data000.cpk` … `data112.cpk`, 6.2 GB, 10,776 files, CRILAYLA entries) of PlatinumGames DAT archives, beside the USM movies and Wwise streams. `cpk.py` reads them.
- **Current:** 1,371 shader files of the engine's older layout (neither Bayonetta 3's shader objects nor Bayonetta 2's BNSH): no magic, `u32 0x88`, a 0x80-byte binding table (0xff = unused), `u32 0xabababab`, `u32` code length, the code area, `u32` control length, the control section (0x880) to the end of the file; the stage is in the code (`shader_split.py` `pg_shader_old`). They are the `.vso` / `.pso` / `.gso` / `.cso` members of five DATs in `data000.cpk`: `shader.dat`, `shader2.dat`, `shadereff.dat`, `shadereffcs.dat`, `autoshadereff.dat`. 1,225 distinct keys, single-stage programs, separate textures and samplers; all build but the four `IndexBuildIndependent_*.gso` geometry shaders, two of which are bound in the prologue.
- A prewarm list recorded on the Mac over the route above (314 lines).

## Rendering

- **Resolved: a magenta title screen** (`tex_flush_upload`, `tex_dirty_upload`, above).
- **Resolved: a green, then darkening picture in the prologue.** NaN in the red and blue channels of the RG11B10F scene target (a NaN written to RG11B10F is stored as `0x7e0`; a negative value is stored as 0, checked with a one-draw Metal test). The source was FXAA sampling its own target through a separate texture bind; see the `feedback_snap` row. GitHub issue #72's NaN-safe min / max did not change it, and neither did compiling the Mac libraries with `-fmetal-math-mode=relaxed`.
- **Current: models black on the iPhone** (issue #163): flight units, ships and buildings render as black silhouettes over a lit sea and sky, and the first on-foot combat scene after the flight is almost entirely black. On the Mac the flight units look dark but their material textures read back correctly (slot 0 albedo: dark panels with orange lights, slot 2: a packed mask map, the BC1 ones: normal maps), so the flight units may be right; the on-foot scene has not been reached on the Mac yet. Ruled out so far: texture uploads of the model textures (`texraw` reads back their data in the plain build), dropped draws (no `Z` lines), the cube-array layer of the IBL probes (the probe-selection buffer is solid `(0, 0, 1)`, index 0).
- **Note:** `DM_NVN_HILITE` overrides did not show on the units with the prewarm list and the pipeline archive on; turn both off for such tests (`DM_PSO_MANIFEST=off`, no `pso_user.tsv`).

## Mac input

The work folder's `run/route.sh` boots a fresh save with `DM_KBD=Player` and presses through the title, a new game (slot 1, Yes, the name, Easy) to the prologue; `run/ctl.sh` appends `DM_CTL` lines (`snap NAME`, `wait S`, presses). Holding `R` shoots in the flight section.

## Found on this game

- **The engine's older shader files in DAT archives inside CPKs** (`pg_shader_old`).
- **Same-pass feedback snapshots for separate texture binds** (`fb_prepare`): [rendering.md](../../../docs/reference/rendering.md#same-pass-feedback-read-through-a-snapshot).

## Open

- The black models of issue #163, above.
- Not run on the iPhone past the first combat scene.
