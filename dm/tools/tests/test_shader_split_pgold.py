"""Exercise shader_split.py's reader of NieR:Automata's older PlatinumGames shader files on synthetic bytes, without game files."""
import os
from pathlib import Path
import struct
import sys
import unittest

os.environ.setdefault('DM_GAME', '0100B8E016F76000')   # shader_split.py loads a profile at import
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'gfx' / 'tools'))
import shader_split as ss


def old_shader(stage_bits):
    """An older PlatinumGames shader file: u32 0x88, binding table, 0xabababab, code length, code area, control length, control section"""
    code = bytearray(0x80 + 32); code[:4] = ss.GFX_MAGIC; struct.pack_into('<I', code, 0x30, 1 | 3 << 5 | stage_bits << 10)
    ctl = bytearray(0x880); ctl[:4] = ss.CTL_MAGIC; struct.pack_into('<I', ctl, 0x20, len(ctl)); struct.pack_into('<I', ctl, 0x704, len(code))
    return struct.pack('<I', 0x88) + b'\xff' * 0x80 + ss.AB4 + struct.pack('<I', len(code)) + bytes(code) + struct.pack('<I', len(ctl)) + bytes(ctl)


class OldShaderFile(unittest.TestCase):
    def test_stage_and_sections(self):
        out = ss.pg_shader_old(old_shader(5), 'x.pso')
        self.assertEqual([(w, st) for w, st, _, _ in out], [('x.pso#PGSO', 'ps')])
        self.assertEqual(len(out[0][2]), 0x880); self.assertEqual(out[0][3][:4], ss.GFX_MAGIC)

    def test_inside_a_dat_through_scan_blob(self):
        f = old_shader(1)
        hdr = b'DAT\0' + struct.pack('<5I', 1, 0x20, 0x24, 0x28, 0x34) + bytes(8)
        tables = struct.pack('<I', 0x40) + b'vso\0' + struct.pack('<I', 8) + b'a.vso\0\0\0' + struct.pack('<I', len(f))
        dat = (hdr + tables).ljust(0x40, b'\0') + f
        out = []; ss.scan_blob(dat, '/d', out)
        self.assertEqual([(w, st) for w, st, _, _ in out], [('/d/a.vso#PGSO', 'vs')])

    def test_other_layouts_are_left_alone(self):
        f = bytearray(old_shader(5)); f[-0x880:-0x87c] = b'\0' * 4   # no control magic where the layout puts it
        self.assertEqual(ss.pg_shader_old(bytes(f), 'x'), [])


if __name__ == '__main__':
    unittest.main()
