#!/bin/bash
# Copy a game's iOS shader libraries to its iPhone app's Documents/<b-chain>.img once, so later installs built with DM_COPY_SHADERS=NO skip them (see docs/development.md, Quick reinstalls).
# They go as one image file (dm/tools/romfs_image.py, the RomFS image format), packed next to the folder and kept while the folder is unchanged: a folder of 55,000 files
# (Breath of the Wild) timed out the phone's file service partway. main.m takes the image only when it is complete (its index comes last) and its inputs.tsv is the app's.
# Usage: push_shaders.sh <device UDID> <bundle ID> <shader library folder, e.g. <id>_build/shader/ios/b-<chain>>
set -eu
[ "$#" = 3 ] || { echo "usage: $0 <device UDID> <bundle ID> <shader library folder>" >&2; exit 1; }
dev=$1 id=$2 src=${3%/}; d=$(basename "$src")
[ -f "$src/inputs.tsv" ] || { echo "no stamp $src/inputs.tsv: not a full dmsh_build.py output" >&2; exit 1; }
python3 -I "$(dirname "$0")/../dm/tools/romfs_image.py" --reuse "$src" "$src.img"
xcrun devicectl device copy to --device "$dev" --domain-type appDataContainer --domain-identifier "$id" --source "$src.img" --destination "Documents/$d.img" -q
echo "shader libraries on the phone: $id ($d.img)"
