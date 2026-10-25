# NieR:Automata The End of YoRHa Edition (`0100B8E016F76000`)

Base game v0 (display version 1.0.1), SDK 12.3.9. PlatinumGames' engine (the Switch port is Virtuos') on NVN. Modules `main`, `subsdk0` (`NintendoSDK_movie`: the logo movies, VP9 in CRI USM through `movie::Decoder`) and `sdk`. CRI's file system (CPK), Wwise sound, Havok, Enlighten. Plays well on the Mac at 30 fps from the boot through the prologue's flight section, the factory shaft and the landing to 2B's first fight on foot (Easy); not run on the iPhone since the models were fixed (GitHub issue #163).

On the Mac (fresh save): the boot, the logo movies, the "booting system" screen, the title, a new game (slot 1, the auto-save notice, a name through swkbd, Easy or Normal), the opening text and the prologue's flight-unit section over the sea and in the clouds (shooting, explosions, enemy bullets, HUD, the damage glitch effect), the factory's side-scrolling and vertical shaft sections, the landing cutscene and 2B's first fight on foot (light / heavy attacks, pod fire, the small machines). 30 fps as on the Switch. About 2.95 GB of the 3.21 GB pool is mapped at boot and stays there.

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
- **Current:** 1,371 shader files of the engine's older layout (neither Bayonetta 3's shader objects nor Bayonetta 2's BNSH): no magic, `u32 0x88`, a 0x80-byte binding table (0xff = unused), `u32 0xabababab`, `u32` code length, the code area, `u32` control length, the control section (0x880) to the end of the file; the stage is in the code (`shader_split.py` `pg_shader_old`). They are the `.vso` / `.pso` / `.gso` / `.cso` members of five DATs in `data000.cpk`: `shader.dat`, `shader2.dat`, `shadereff.dat`, `shadereffcs.dat`, `autoshadereff.dat`. 1,225 distinct keys, single-stage programs, separate textures and samplers; all build, the four `IndexBuildIndependent_*.gso` geometry shaders as compute kernels (below).
- A prewarm list recorded on the Mac over the route above (314 lines).

## Rendering

- **Resolved: a magenta title screen** (`tex_flush_upload`, `tex_dirty_upload`, above).
- **Resolved: a green, then darkening picture in the prologue.** NaN in the red and blue channels of the RG11B10F scene target (a NaN written to RG11B10F is stored as `0x7e0`; a negative value is stored as 0, checked with a one-draw Metal test). The source was FXAA sampling its own target through a separate texture bind; see the `feedback_snap` row. GitHub issue #72's NaN-safe min / max did not change it, and neither did compiling the Mac libraries with `-fmetal-math-mode=relaxed`.
- **Resolved: models black, then missing** (issue #163). The model draws (`ps_fp_pbs00` `176547245fdff6b1`, `ps_cns` `57290de615ee7295`, …) are indexed indirect draws (`vc=0` in the pass dump) that test depth for equality and write none. The depth they test is the depth prepass, drawn one mesh at a time with a zmap vertex shader (`zmap_vs_*`, e.g. the skinned `eaf2df477c4919a3`) and one of four geometry shaders, `IndexBuildIndependent_{b,x}{cw,ccw}.gso` (`x*`: inputs at locations 2 / 3 / 4, `b*`: 3 / 4 / 5). They are passthrough shaders: each triangle goes on to the rasterizer unless the shader culls it (frustum, back face, no pixel centre covered) with a viewport mask of 0, and each kept triangle's indices (the vertex shader passes `gl_VertexIndex` on) go into the mesh's index list (`gp_s1`) and its count (`gp_s0`, an atomic add) for the shading pass. On the way:
  - the shaders did not load (no slot for their storage buffers): counts stayed 0, so the prepass drew the meshes unculled through the vertex shader alone and the shading drew nothing (black silhouettes under the fog). Fixed by the geometry stage's storage-buffer table at c0 `0x410` and a reader library without the viewport-mask copy;
  - loaded, they emitted nothing (the translator answered "passthrough supported"), so the prepass vanished too and with it the units. Fixed: [Passthrough geometry shaders](../../../docs/reference/shaders.md#passthrough-geometry-shaders);
  - the geometry replay then ran at 5 fps (a GPU wait before every indexed draw, a one-thread compaction, four encoders and a pass split per mesh). Fixed: [Geometry-shader draws encoded a run at a time](../../../docs/reference/rendering.md#geometry-shader-draws-encoded-a-run-at-a-time). The flight section now takes about 14 ms of GPU and 14 ms of submit a frame (frame time p50 33.3 ms).
- **Note:** `DM_NVN_HILITE` overrides did not show on the units with the prewarm list and the pipeline archive on; turn both off for such tests (`DM_PSO_MANIFEST=off`, no `pso_user.tsv`).

## Mac input

The work folder's `run/route.sh` boots a fresh save with `DM_KBD=Player` and presses through the title, a new game (slot 1, Yes, the name, Easy) to the prologue; `run/cont.sh` loads slot 1 from the title (Continue) into the flight section; `run/ctl.sh` appends `DM_CTL` lines (`snap NAME`, `wait S`, presses). Holding `R` shoots in the flight section.

## Found on this game

- **The engine's older shader files in DAT archives inside CPKs** (`pg_shader_old`).
- **Same-pass feedback snapshots for separate texture binds** (`fb_prepare`): [rendering.md](../../../docs/reference/rendering.md#same-pass-feedback-read-through-a-snapshot).
- **Passthrough geometry shaders**: [shaders.md](../../../docs/reference/shaders.md#passthrough-geometry-shaders).
- **Geometry-shader draws encoded a run at a time**, with the parallel index compaction: [rendering.md](../../../docs/reference/rendering.md#geometry-shader-draws-encoded-a-run-at-a-time).

## Open

- Not run on the iPhone since the models were fixed (issue #163), nor past the first combat scene.
