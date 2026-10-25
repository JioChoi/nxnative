// Primitive assembly for geometry compute. Values are capture-record indices;
// restart separates runs before strips/fans are assembled. No game-specific state.
#pragma once
#include <stdint.h>
#include <stddef.h>

static int nvn_geo_topology(int prim) {
    switch (prim) {
        case 0: return 0;
        case 1: case 2: case 3: return 1;
        case 10: case 11: return 2;
        case 4: case 5: case 6: case 7: return 3;
        case 12: return 4;
        default: return -1;
    }
}
static int nvn_geo_width(int topology) { static const int width[] = {1, 2, 4, 3, 6}; return topology >= 0 && topology < 5 ? width[topology] : 0; }

// Returns words required/written. SIZE_MAX means unsupported topology or overflow.
// dst == NULL counts only. A short destination is never partially written.
static size_t nvn_geo_assemble(int prim, const uint32_t *src, size_t count, int restart, uint32_t restart_index, uint32_t *dst, size_t capacity) {
    int width = nvn_geo_width(nvn_geo_topology(prim));
    if (!width) return SIZE_MAX;
    if (dst) {
        size_t needed = nvn_geo_assemble(prim, src, count, restart, restart_index, NULL, 0);
        if (needed == SIZE_MAX || needed > capacity) return SIZE_MAX;
    }
    size_t words = 0;
    for (size_t begin = 0; begin < count;) {
        size_t end = begin;
        while (end < count && !(restart && src[end] == restart_index)) end++;
        size_t n = end - begin, np = 0;
        switch (prim) {
            case 0: np = n; break;
            case 1: np = n / 2; break;
            case 2: np = n > 1 ? n : 0; break;
            case 3: np = n > 1 ? n - 1 : 0; break;
            case 4: np = n / 3; break;
            case 5: case 6: np = n > 2 ? n - 2 : 0; break;
            case 7: np = n / 4 * 2; break;
            case 10: np = n / 4; break;
            case 11: np = n > 3 ? n - 3 : 0; break;
            case 12: np = n / 6; break;
        }
        if (np > (SIZE_MAX - words) / (size_t)width) return SIZE_MAX;
        if (dst && words + np * width > capacity) return SIZE_MAX;
        for (size_t i = 0; dst && i < np; i++) {
            size_t k[6] = {0};
            switch (prim) {
                case 0: k[0] = i; break;
                case 1: k[0] = 2*i; k[1] = 2*i+1; break;
                case 2: k[0] = i; k[1] = (i+1)%n; break;
                case 3: k[0] = i; k[1] = i+1; break;
                case 4: for (int c=0;c<3;c++) k[c]=3*i+c; break;
                case 5: k[0]=i+(i&1); k[1]=i+1-(i&1); k[2]=i+2; break;
                case 6: k[0]=0; k[1]=i+1; k[2]=i+2; break;
                case 7: k[0]=4*(i/2); k[1]=k[0]+1+(i&1); k[2]=k[1]+1; break;
                case 10: for (int c=0;c<4;c++) k[c]=4*i+c; break;
                case 11: for (int c=0;c<4;c++) k[c]=i+c; break;
                case 12: for (int c=0;c<6;c++) k[c]=6*i+c; break;
            }
            for (int c=0;c<width;c++) dst[words+i*width+c]=src[begin+k[c]];
        }
        words += np * width;
        begin = end < count ? end + 1 : end;
    }
    return words;
}

// One ordered compaction per draw. Output is a point/line/triangle list, so it
// does not depend on Metal primitive-restart behaviour. Preserve triangle-strip
// winding and the last vertex. Unwritten GS index slots are UINT_MAX.
// One threadgroup: each thread counts a run of GS groups, the runs' offsets are
// scanned in threadgroup memory, then each thread writes its run in order. A single
// thread doing it all took 120 of 123 ms of NieR:Automata's flight frame on an M-series Mac.
static const char k_nvn_geo_pack_msl[] =
"#include <metal_stdlib>\nusing namespace metal;\n"
"struct geo_pack { device const uint *src; device uint *dst; device uint *args; uint groups, stride, topology, capacity; };\n"
"static uint geo_group(constant geo_pack& j, uint g, uint o, bool w) { uint a=0,b=0,n=0;\n"
" for(uint i=0;i<j.stride;i++) { uint v=j.src[g*j.stride+i];\n"
"  if(v==0xffffffffu) {n=0;continue;}\n"
"  if(j.topology==1) {if(w && o<j.capacity) j.dst[o]=v; o++;}\n"
"  else if(j.topology==6) {if(n) {if(w && o+2<=j.capacity) {j.dst[o]=b;j.dst[o+1]=v;} o+=2;}}\n"
"  else if(n>=2) {if(w && o+3<=j.capacity) {j.dst[o]=(n&1)?b:a;j.dst[o+1]=(n&1)?a:b;j.dst[o+2]=v;} o+=3;}\n"
"  a=b;b=v;n++;\n"
" }\n"
" return o;\n"
"}\n"
"kernel void geo_pack_indices(constant geo_pack& j [[buffer(0)]], uint t [[thread_position_in_threadgroup]], uint nt [[threads_per_threadgroup]]) {\n"
" threadgroup uint base[1025];\n"
" uint per=(j.groups+nt-1)/nt, g0=min(t*per,j.groups), g1=min(g0+per,j.groups), o=0;\n"
" for(uint g=g0;g<g1;g++) o=geo_group(j,g,o,false);\n"
" base[t]=o; threadgroup_barrier(mem_flags::mem_threadgroup);\n"
" if(t==0) {uint s=0; for(uint i=0;i<nt;i++) {uint c=base[i]; base[i]=s; s+=c;} base[nt]=s;}\n"
" threadgroup_barrier(mem_flags::mem_threadgroup);\n"
" o=base[t]; for(uint g=g0;g<g1;g++) o=geo_group(j,g,o,true);\n"
" if(t==0) {j.args[0]=min(base[nt],j.capacity);j.args[1]=1;j.args[2]=0;j.args[3]=0;j.args[4]=0;}\n"
"}\n";
