// General geometry replay: native VS capture -> GS compute -> ordered index
// compaction -> ordinary rendering. Included after the resource/pipeline helpers.
static NvProg *g_geo_program;
static uint64_t g_geo_indices, g_geo_index_bytes, g_geo_args;
static const p_draw *g_geo_vrange; static uint32_t g_geo_count;   // a lean raster draw: the vertices its indices name, for vertex conversion (vconv_bind)
// A lean geometry draw split in two: its capture, kernel and compaction (prepared), and its raster, which the program's own stages draw.
// geo_hoist_scan (nvn_mtl.m) prepares the geometry draws that follow one, before the render pass opens, so that their rasters share
// one pass instead of each ending and reopening it; replay then meets those ops and draws only the raster (geo_draw)
typedef struct { const uint8_t *op; uint64_t sub; NvShader *vs; uint64_t packed, packed_size, args; p_draw vrange; uint32_t first_instance; int topology, discard; uint32_t count; } geo_ready;   // count: a direct draw of that many indices (geo_lean_tri), else indirect
enum { GEO_READY_MAX = 512 };   // NieR:Automata's heaviest scenes have about 465 geometry draws a frame
static geo_ready g_geo_ready[GEO_READY_MAX]; static int g_geo_ready_n;
static geo_ready *g_geo_prep_out;   // set: geo_draw_ prepares a lean draw into it and draws nothing (op stays NULL when it cannot)
static uint64_t g_geo_hoisted, g_geo_hoist_runs, g_geo_hoist_stop[6];
static void geo_hoist_scan(void);
static int geo_hoist_on(void) { static int m = -1; if (m < 0) { const char *e = getenv("DM_NVN_GEO_HOIST"); m = !(e && !strcmp(e, "0")); } return m; }
static void draw(const p_draw *dw, const p_draw_idx *di);
static void lc_land(void);

static int geo_cmp_index(const void *a, const void *b) {
    uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
    return (x > y) - (x < y);
}
enum { GEO_CAPTURE_VERTICES = (256u << 20) / (144 * 4) };
// The capture span is already bounded by its scratch allocation. A bit per
// vertex gives the same ascending unique list without sorting every reference.
// Keep an in-place sort fallback for allocation failure or a wider caller.
// order may be changed; a short destination is never partially written.
static size_t geo_capture_unique(uint32_t *order, size_t count, int restart, uint32_t restart_index,
                                 uint32_t lo, uint64_t span, uint32_t *dst, size_t capacity) {
    if (!span || span > (uint64_t)UINT32_MAX + 1 - lo) return SIZE_MAX;
    for (size_t i = 0; i < count; i++) if (!(restart && order[i] == restart_index) &&
        (order[i] < lo || (uint64_t)order[i] - lo >= span)) return SIZE_MAX;
    size_t nw = (size_t)((span + 63) / 64), n = 0;
    uint64_t *bits = span <= GEO_CAPTURE_VERTICES ? calloc(nw, sizeof *bits) : NULL;
    if (bits) {
        for (size_t i = 0; i < count; i++) if (!(restart && order[i] == restart_index)) {
            uint32_t v = order[i] - lo; bits[v >> 6] |= 1ull << (v & 63);
        }
        for (size_t i = 0; i < nw; i++) n += (size_t)__builtin_popcountll(bits[i]);
        if (n <= capacity) for (size_t i = 0, k = 0; i < nw; i++) {
            uint64_t word = bits[i];
            while (word) { dst[k++] = lo + (uint32_t)(i * 64 + __builtin_ctzll(word)); word &= word - 1; }
        }
        free(bits);
    } else {
        size_t references = 0;
        for (size_t i = 0; i < count; i++) if (!(restart && order[i] == restart_index)) order[references++] = order[i];
        qsort(order, references, sizeof *order, geo_cmp_index);
        for (size_t i = 0; i < references; i++) if (!i || order[i] != order[i - 1]) n++;
        if (n <= capacity) for (size_t i = 0, k = 0; i < references; i++)
            if (!i || order[i] != order[i - 1]) dst[k++] = order[i];
    }
    return n <= capacity ? n : SIZE_MAX;
}
// Only CPU reads need a completed snapshot. Native vertex capture stays on the
// GPU and follows earlier writes through normal encoder ordering. Unknown or
// overflowing write sets always retain the conservative drain.
static int geo_cpu_hazard(uint64_t lo, uint64_t n) {
    if (!n) return 0;
    if (!lo || n > UINT64_MAX - lo || g_sbw_over) return 1;
    return pend_hits(lo, n) || sbw_hits(lo, lo + n) || gw_live(lo, lo + n);
}
static int geo_conversion_hazard(NvShader *vs) {
    for (int v = 0; v < vs->rf.nv; v++) {
        int a = vs->rf.v[v].nvn;
        if (a >= S.nattr || !vconv_wanted(&vs->rf, v, NULL, 0)) continue;
        int st = S.attr[a].stream;
        if (st < 16 && S.vbuf[st].address && geo_cpu_hazard(S.vbuf[st].address, S.vbuf[st].size)) return 1;
    }
    return 0;
}
// GS output -> counts -> offsets/indirect args -> packed indices stay in one
// compute encoder. Each transition needs dispatch ordering and device-memory
// visibility; full queue barriers remain at the encoder's boundaries.
static void geo_pack_encode(id<MTL4ComputeCommandEncoder> ce, id<MTLComputePipelineState> __strong *pack, uint64_t job, uint32_t groups) {
    enc_c_args(ce, job, 0);
    for (int i = 0; i < 3; i++) {
        [ce barrierAfterEncoderStages:MTLStageDispatch beforeEncoderStages:MTLStageDispatch visibilityOptions:MTL4VisibilityOptionDevice];
        enc_c_pso(ce, pack[i]);
        NSUInteger threads = i == 1 ? MIN((NSUInteger)256, pack[i].maxTotalThreadsPerThreadgroup) : 64;
        enc_c_dispatch(ce, i == 1 ? MTLSizeMake(1, 1, 1) : MTLSizeMake((groups + 63) / 64, 1, 1), MTLSizeMake(threads, 1, 1));
    }
}
static id<MTLRenderPipelineState> geo_capture_pso(NvShader *vs) {
    pso_key k; static const uint32_t cf[8]; pso_key_fill(&k, vs, nil, cf, 0, 0, 1, 0); k.fsfn = GEO_MARK;
    id<MTLRenderPipelineState> ps = hm_get_id(&g_pso, &k);
    if (ps || hm_get_id(&g_pso_fail, &k)) return ps;
    NSError *e = nil;
    ps = [g_dev newRenderPipelineStateWithDescriptor:pso_desc(&k, vs, nil, nil, 0) error:&e];
    if (ps) hm_put_id(&g_pso, &k, ps);
    else { hm_put_id(&g_pso_fail, &k, NSNull.null); MLOG("[RED] geometry vertex capture %s: %s", vs->key, e.localizedDescription.UTF8String); }
    return ps;
}
// One diagnostic snapshot through the same native vertex fetch as geometry
// capture. Refuse shaders with writes: observing them must not run their side
// effects twice. Records retain the documented 144-word geometry capture ABI.
static void vertex_dump(NvShader *vs, const p_draw *dw, const p_draw_idx *di) {
    static const char *key, *path; static int configured, captured, limit; static uint64_t frame;
    if (!configured) { key = getenv("DM_NVN_VSDUMP_KEY"); path = getenv("DM_NVN_VSDUMP");
        const char *f = getenv("DM_NVN_VSDUMP_FROM"), *n = getenv("DM_NVN_VSDUMP_MAX");
        frame = f ? strtoull(f,NULL,0) : 0; limit = n ? atoi(n) : key && !strcmp(key,"*") ? 256 : 1; configured = 1; }
    int all = key && !strcmp(key,"*");
    if (captured >= limit || !key || !path || atomic_load(&g_cur_frame) < frame || !vs || (!all && strncmp(vs->key, key, 16)) || !vs->geo_fn ||
        (dw ? dw->ind || dw->mdi_cnt : di->ind || di->mdi_cnt) || S.rast_discard) return;
    for (int i = 0; i < vs->rf.nb; i++) if (vs->rf.b[i].rw) return;
    int count = dw ? dw->count : di->count, instances = dw ? dw->instances : di->instances;
    if (count <= 0 || instances > 1 || (di && (di->itype < 0 || di->itype > 2))) return;
    unsigned number = captured++; lc_land(); rp_flush(); mdi_wait();
    uint32_t lo = dw ? (uint32_t)dw->first : UINT32_MAX, hi = dw ? lo + count - 1 : 0;
    if (di) {
        int size = 1 << di->itype;
        if (!nvn_host_readable(di->ibuf, (uint64_t)count * size)) return;
        for (int i = 0; i < count; i++) { uint32_t n = 0; memcpy(&n, (void *)(uintptr_t)(di->ibuf + (uint64_t)i * size), size);
            if (S.restart.on && n == (S.restart.index & (size == 1 ? 255u : size == 2 ? 65535u : UINT32_MAX))) continue;
            lo = MIN(lo, n); hi = MAX(hi, n); }
        if (lo == UINT32_MAX || (int64_t)lo + di->base_vertex < 0 || (int64_t)hi + di->base_vertex > UINT32_MAX) return;
        lo += di->base_vertex; hi += di->base_vertex;
    }
    uint64_t span = (uint64_t)hi - lo + 1, bytes = span * 144 * 4;
    if (!span || bytes > (64u << 20)) return;
    id<MTLRenderPipelineState> ps = geo_capture_pso(vs); if (!ps) return;
    id<MTLBuffer> output; size_t offset = arena_alloc(bytes + 512, &output);
    uint64_t gpu = output.gpuAddress + offset;
    uint64_t info[4] = {gpu + 512, 0, (uint64_t)lo | span << 32, (uint32_t)(dw ? dw->base_instance : di->base_instance) | 1ull << 32};
    nvn_geo_capture gc = {info[0], 0, {(uint32_t)lo, (uint32_t)span, (uint32_t)info[3], 1}, GEO_CAPTURE_WORDS, {0}};
    for (int k = 0; k < GEO_CAPTURE_WORDS; k++) gc.slot[k] = (int16_t)k;   // every word, in order
    memcpy((uint8_t *)output.contents + offset, &gc, sizeof gc);
    vbuf_prepare();
    if (g_blit) { enc_b_end(g_blit); g_blit = nil; }
    MTL4RenderPassDescriptor *rd = [MTL4RenderPassDescriptor new]; rd.renderTargetWidth = rd.renderTargetHeight = rd.defaultRasterSampleCount = 1;
    id<MTL4RenderCommandEncoder> en = enc_r_begin(rd); enc_r_pso(en, ps); bind_vbufs(en);
    p_draw range = {.first = (int32_t)lo, .count = (int32_t)span, .instances = 1, .base_instance = (int32_t)(info[3] & UINT32_MAX)};
    vconv_bind(en, vs, 1, &range, NULL); args_dirty_all(); encode_args(en, vs, 0, 1); enc_r_vbuf(en, 13, gpu);
    enc_r_draw_instanced(en, MTLPrimitiveTypePoint, lo, span, 1, (uint32_t)info[3]); enc_r_end(en);
    mdi_wait();
    char filename[1200]; if (all) snprintf(filename,sizeof filename,"%s_%04u_%s.bin",path,number,vs->key); else strlcpy(filename,path,sizeof filename);
    FILE *f = fopen(filename, "wb"); if (f) { uint32_t header[4] = {lo, (uint32_t)span, all ? 4u : 144u, (uint32_t)info[3]}; fwrite(header, sizeof header, 1, f);
        if (all) for (uint64_t i=0;i<span;i++) fwrite((uint8_t *)output.contents + offset + 512 + i*144*4,1,16,f);
        else fwrite((uint8_t *)output.contents + offset + 512, 1, bytes, f); fclose(f); }
    MLOG("vertex dump %s: first %u, count %llu, %s", vs->key, lo, (unsigned long long)span, filename);
    args_dirty_all(); g_enc_pso = nil; g_enc_dss = nil; g_ed.ok = 0;
}
static void geo_draw(NvProg *pr, const p_draw *dw0, const p_draw_idx *di0);
// A draw whose capture would need more scratch than the bound, drawn as smaller draws in the hardware's order: one instance at a time, then runs of a list topology's primitives
// (points, lines, triangles, lines and triangles with adjacency), each with its own index range. The capture lays the vertices out by vertex number from the lowest index to the highest for
// every instance, and the geometry shader's output takes room per primitive: OCTOPATH TRAVELER II draws 129,032 triangles of one mesh through a cube shadow map's geometry shader in its
// first battle, which asked for more than the bound and was skipped. The index list is read once and cut into runs of at most GEO_RUN_PRIMS primitives whose index range takes at most
// GEO_RUN_SPAN of capture; the runs skip the GPU wait their draw would take (the outer draw waited already, and nothing in between writes index or vertex data). Halving the list at every
// level instead, each part reading its indices again and waiting for the GPU, made that one frame take 805 ms. Strips, fans and primitive restart cannot be cut that way. The draw's
// primitive numbers restart at 0 in each run: its geometry shader (cb17a9453d4789d0, Unreal's one-pass point light shadow) reads only its index map by them, but one that read its
// primitive ID would see other numbers than on the Switch (such draws were skipped before).
// Returns 0 when the draw cannot be split (the caller reports it)
enum { GEO_RUN_PRIMS = 4096 }; static const uint64_t GEO_RUN_SPAN = 64ull << 20;
static int g_geo_split_depth;   // > 0 while geo_split draws its runs: they skip the GPU wait
// A geometry draw's capture and compute, encoded later by geo_flush: a run of draws prepared ahead of their pass (geo_hoist_scan) shares one
// capture encoder and one compute encoder instead of two encoders each. Each job keeps the state its draw had, for its bindings.
typedef struct { void *state; id<MTLRenderPipelineState> capture; NvShader *vs, *kernel; int lean, tri, indexed, instances; uint32_t record, first_instance, unique;
                 int32_t base_vertex, first, count; uint64_t capture_info, capture_ix, capture_ix_bytes, binds[5], np, groups, job, ljob, packed_size; p_draw vrange; } geo_job;
static geo_job g_geo_jobs[GEO_READY_MAX + 1]; static int g_geo_njobs;
static id<MTLComputePipelineState> g_geo_pack[3], g_geo_lean_k[3];   // geo_pack_count, geo_pack_scan, geo_pack_write; geo_lean_cull, geo_lean_index, geo_lean_tri (nvn_geometry.h)
static void geo_flush(void) {
    if (!g_geo_njobs) return;
    static typeof(S) now; memcpy(&now, &S, sizeof S);
    if (g_blit) { enc_b_end(g_blit); g_blit = nil; }
    id<MTL4RenderCommandEncoder> re = nil;
    for (int i = 0; i < g_geo_njobs; i++) { geo_job *j = &g_geo_jobs[i];
        if (!j->record) continue;   // a lean kernel that reads no capture word needs no capture
        if (!re) { MTL4RenderPassDescriptor *rd = [MTL4RenderPassDescriptor new]; rd.renderTargetWidth = rd.renderTargetHeight = rd.defaultRasterSampleCount = 1; re = enc_r_begin(rd); }
        memcpy(&S, j->state, sizeof S);
        enc_r_pso(re, j->capture); bind_vbufs(re);
        vconv_bind(re, j->vs, 1, &j->vrange, NULL);
        args_dirty_all(); encode_args(re, j->vs, 0, 1); enc_r_vbuf(re, 13, j->capture_info);
        if (j->indexed) enc_r_draw_indexed(re, MTLPrimitiveTypePoint, j->unique, MTLIndexTypeUInt32, j->capture_ix, j->capture_ix_bytes, j->instances, j->base_vertex, j->first_instance);
        else enc_r_draw_instanced(re, MTLPrimitiveTypePoint, j->first, j->count, j->instances, j->first_instance);
    }
    if (re) enc_r_end(re);
    id<MTL4ComputeCommandEncoder> ce = enc_c_begin(NULL);
    for (int i = 0; i < g_geo_njobs; i++) { geo_job *j = &g_geo_jobs[i];
        memcpy(&S, j->state, sizeof S); memcpy(g_geo_bind, j->binds, sizeof j->binds);
        if (i) [ce barrierAfterEncoderStages:MTLStageDispatch beforeEncoderStages:MTLStageDispatch visibilityOptions:MTL4VisibilityOptionDevice];   // kernels in draw order (their storage appends)
        args_dirty_all(); encode_args(ce, j->kernel, 2, 0); enc_c_pso(ce, j->kernel->geo_pso);
        enc_c_dispatch(ce, MTLSizeMake((j->np + 31) / 32, j->instances, j->kernel->geo_invocations), MTLSizeMake(32, 1, 1));
        if (j->tri) {   // triangle slots straight from the kernel's output: no compaction
            enc_c_args(ce, j->ljob, 0);
            [ce barrierAfterEncoderStages:MTLStageDispatch beforeEncoderStages:MTLStageDispatch visibilityOptions:MTL4VisibilityOptionDevice];
            enc_c_pso(ce, g_geo_lean_k[2]); enc_c_dispatch(ce, MTLSizeMake((j->groups + 63) / 64, 1, 1), MTLSizeMake(64, 1, 1));
            free(j->state); j->state = NULL; j->capture = nil; j->vs = j->kernel = nil;
            continue;
        }
        if (j->lean) {
            enc_c_args(ce, j->ljob, 0);
            [ce barrierAfterEncoderStages:MTLStageDispatch beforeEncoderStages:MTLStageDispatch visibilityOptions:MTL4VisibilityOptionDevice];
            enc_c_pso(ce, g_geo_lean_k[0]); enc_c_dispatch(ce, MTLSizeMake((j->groups + 63) / 64, 1, 1), MTLSizeMake(64, 1, 1));
        }
        geo_pack_encode(ce, g_geo_pack, j->job, (uint32_t)j->groups);
        if (j->lean) {
            enc_c_args(ce, j->ljob, 0);
            [ce barrierAfterEncoderStages:MTLStageDispatch beforeEncoderStages:MTLStageDispatch visibilityOptions:MTL4VisibilityOptionDevice];
            enc_c_pso(ce, g_geo_lean_k[1]); enc_c_dispatch(ce, MTLSizeMake((j->packed_size / 4 + 63) / 64, 1, 1), MTLSizeMake(64, 1, 1));
        }
        free(j->state); j->state = NULL; j->capture = nil; j->vs = j->kernel = nil;
    }
    enc_c_end(ce);
    g_geo_njobs = 0;
    memcpy(&S, &now, sizeof S); args_dirty_all(); g_enc_pso = nil; g_enc_dss = nil; g_ed.ok = 0;
}
static int geo_sync_cpu(int *drained) {
    if (*drained) return 1;
    geo_flush();   // prepared kernels may write what the CPU is about to read
    mdi_wait();
    if (g_mdi_ev.signaledValue < g_mdi_v) return 0;
    memset(g_vconv, 0, sizeof g_vconv); *drained = 1;
    return 1;
}
static int geo_split(NvProg *pr, const p_draw *dw, const p_draw_idx *di, int count, int instances, uint32_t first_instance) {
    int prim = dw ? dw->prim : di->prim, per = prim == 0 ? 1 : prim == 1 ? 2 : prim == 4 ? 3 : prim == 10 ? 4 : prim == 12 ? 6 : 0;
    if (!per || (di && S.restart.on) || count / per < 1) return 0;
    static uint64_t nsplit; if (nsplit++ < 4) MLOG("geometry draw above the scratch bound: drawn in parts (vertex %s, geometry %s, %d primitives, %d instances)",
                                                   pr->sh[0] ? pr->sh[0]->key : "-", pr->sh[2] ? pr->sh[2]->key : "-", count / per, instances);
    p_draw a = dw ? *dw : (p_draw){0}; p_draw_idx b = di ? *di : (p_draw_idx){0};
    g_geo_split_depth++;
    if (instances > 1) {
        for (int i = 0; i < instances; i++) {
            if (dw) { a.instances = 1; a.base_instance = (int32_t)(first_instance + i); geo_draw(pr, &a, NULL); }
            else { b.instances = 1; b.base_instance = (int32_t)(first_instance + i); geo_draw(pr, NULL, &b); }
        }
        g_geo_split_depth--; return 1;
    }
    int np = count / per, isz = di ? 1 << di->itype : 0;
    for (int p0 = 0; p0 < np;) {
        int p1 = p0; uint32_t lo = UINT32_MAX, hi = 0;
        while (p1 < np && p1 - p0 < GEO_RUN_PRIMS) {   // extend the run while its index range fits
            uint32_t l = lo, h = hi;
            for (int k = 0; k < per; k++) {
                uint32_t v = (uint32_t)(p1 * per + k);
                if (di) { v = 0; memcpy(&v, (const uint8_t *)(uintptr_t)di->ibuf + (size_t)(p1 * per + k) * isz, isz); }
                l = MIN(l, v); h = MAX(h, v);
            }
            if (p1 > p0 && ((uint64_t)h - l + 1) * 144 * 4 > GEO_RUN_SPAN) break;
            lo = l; hi = h; p1++;
        }
        if (p0 == 0 && p1 == np) {   // one run is the whole draw: its own outputs are what exceed the bound, so halve it
            p1 = np / 2; if (!p1) { g_geo_split_depth--; return 0; }
        }
        if (dw) { a.first = dw->first + p0 * per; a.count = (p1 - p0) * per; geo_draw(pr, &a, NULL); }
        else { b.ibuf = di->ibuf + ((uint64_t)(p0 * per) << di->itype); b.count = (p1 - p0) * per; geo_draw(pr, NULL, &b); }
        p0 = p1;
    }
    g_geo_split_depth--;
    return 1;
}
static void geo_raster(NvProg *pr, const geo_ready *g) {
    if (g->discard || !g->vs) return;
    NvProg *raster = pr->geo_lean_raster;
    if (!raster || raster->sh[0] != g->vs || raster->sh[1] != pr->sh[1]) { raster = [NvProg new]; raster->sh[0] = g->vs; raster->sh[1] = pr->sh[1]; pr->geo_lean_raster = raster; }
    g_geo_program = raster; g_geo_indices = g->packed; g_geo_index_bytes = g->packed_size; g_geo_args = g->args; g_geo_vrange = &g->vrange; g_geo_count = g->count;
    p_draw synthetic = {.prim = g->topology == 0 ? 0 : g->topology == 1 ? 1 : 4, .count = 3, .instances = 1, .base_instance = (int32_t)g->first_instance};
    draw(&synthetic, NULL);
    g_geo_program = nil; g_geo_vrange = NULL; g_geo_count = 0; args_dirty_all();
}
// DM_NVN_GEO_LEAN_CHECK=<draws>[@<frame>] (diagnostic): for that many lean-eligible draws (from that frame on), the full kernel and the lean one each run
// without drawing, the GPU is drained, and the CPU checks that each full output vertex is input vertex map[v] (same position)
// and that the lean index list is what the full kernel's reader would draw. Their storage writes happen three times, so
// those frames can differ.
static struct { int on, width; uint32_t os, first; const uint32_t *raw, *out, *packed, *args, *map, *input; uint32_t count; } g_geo_ck[3];   // count: geo_lean_tri's slots, else 0
static int g_geo_ck_mode;   // 1: full kernel, 2: lean kernel, recorded instead of drawn
static void geo_check_compare(void) {
    const typeof(g_geo_ck[0]) *f = &g_geo_ck[1], *l = &g_geo_ck[2];
    uint32_t n = f->args[0], w = (uint32_t)f->width, kept = 0; size_t pos_bad = 0, mixed = 0;
    uint32_t *want = malloc((size_t)n * 4 + 4);
    if (!want) return;
    for (uint32_t i = 0; i + w <= n; i += w) {
        uint32_t m0 = f->out[(size_t)f->packed[i] * f->os + 5] & 1;
        for (uint32_t k = 0; k < w; k++) { uint32_t v = f->packed[i + k];
            mixed += (f->out[(size_t)v * f->os + 5] & 1) != m0;
            for (int c = 0; c < 4; c++) pos_bad += f->out[(size_t)v * f->os + c] != f->input[(size_t)f->map[v] * GEO_CAPTURE_WORDS + c]; }
        if (m0) for (uint32_t k = 0; k < w; k++) want[kept++] = f->map[f->packed[i + k]] + f->first;
    }
    uint32_t got_n = l->args[0]; const uint32_t *got = l->packed; uint32_t *tri = NULL;
    if (l->count) {   // geo_lean_tri: the kept triangles, without those collapsed onto one vertex
        tri = malloc((size_t)l->count * 4 + 4); got_n = 0;
        for (uint32_t i = 0; tri && i + 3 <= l->count; i += 3) if (!(l->packed[i] == l->packed[i + 1] && l->packed[i] == l->packed[i + 2])) for (int k = 0; k < 3; k++) tri[got_n++] = l->packed[i + k];
        got = tri;
    }
    int same = got && got_n == kept && !memcmp(want, got, (size_t)kept * 4);
    free(tri);
    free(want);
    static long n_ok, n_bad; same && !pos_bad ? n_ok++ : n_bad++;
    MLOG("[geo-check] full: %u indices, %u kept, %zu position mismatches, %zu mixed masks; lean: %u indices, %s (%ld matched, %ld not so far)",
         n, kept, pos_bad, mixed, got_n, same ? "same list" : "[RED] different list", n_ok, n_bad);
}
// A passthrough geometry shader cannot change the vertices it passes on: with input of points, lines or triangles, its lean
// kernel (dmsh_geometry.lean) reads only the capture words it decides with, writes the viewport masks and its storage, and the
// program's own vertex and fragment stages draw the kept primitives. Instanced or multi-invocation draws, vertex shaders with
// storage writes (they would run twice) and transform feedback keep the full kernel and its reader. DM_NVN_GEO_LEAN=0 turns it off.
static NvShader *geo_lean_pick(NvProg *pr, NvShader *gs, int topology, int instances) {
    static int lean_on = -1; if (lean_on < 0) { const char *e = getenv("DM_NVN_GEO_LEAN"); lean_on = !(e && !strcmp(e, "0")); }
    NvShader *vs = pr->sh[0], *base = pr->sh[2];
    if (!lean_on || instances > 1 || (gs && gs->geo_invocations != 1) || base->geo_invocations != 1 || !vs || vs->discard_fn || S.tf_on || !pr->sh[1] || !(topology == 0 || topology == 1 || topology == 3)) return NULL;
    if (!(base->geo_lean_tried >> topology & 1)) {
        base->geo_lean_tried |= 1 << topology;
        NSString *key = [[NSString stringWithUTF8String:base->key] substringToIndex:16];
        base->geo_lean_v[topology] = shader_get([key stringByAppendingFormat:@"_bl%d", topology], 2);
    }
    NvShader *lean = base->geo_lean_v[topology];
    if (!lean || !lean->geo_slot || lean->geo_vertices != (uint32_t)nvn_geo_width(topology) || lean->geo_invocations != 1) return NULL;
    if (!lean->geo_pso && !lean->geo_pso_failed) { NSError *e = nil; lean->geo_pso = [g_dev newComputePipelineStateWithFunction:lean->fn error:&e];
        if (!lean->geo_pso || lean->geo_pso.maxTotalThreadsPerThreadgroup < 32) { lean->geo_pso = nil; lean->geo_pso_failed = 1; MLOG("[RED] lean geometry compute %s: %s; the full kernel draws", lean->key, e ? e.localizedDescription.UTF8String : "no 32-thread group"); } }
    return lean->geo_pso ? lean : NULL;
}
// Passthrough geometry as mesh draws: a lean triangle-list draw whose pair has a mesh library, <vertex key>_<kernel key>_m3 (dm/gfx/tools/dmsh_mesh.py, built for the
// pairs in the profile's geo_mesh.tsv), is one mesh draw inside the render pass. A threadgroup per 32 triangles runs the pulling vertex shader per index and the kernel body
// per triangle, and culls what the kernel culls: no capture, no compute, no index compaction, no pass break. Its pipeline is built in the background; until it is ready, and
// for every other pair, the compute path draws. DM_NVN_GEO_MESH=0 turns it off
static int geo_mesh_on(void) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("DM_NVN_GEO_MESH"); on = !(e && !strcmp(e, "0")); if (!on) MLOG("DM_NVN_GEO_MESH=0: passthrough geometry draws through the compute path"); }
    return on;
}
enum { GEO_MESH_PRIMS = 32, GEO_MESH_DRAW_SLOT = 12, GEO_MESH_GS_SLOT = 13 };   // = dmsh_mesh.py PRIMS, DRAW_SLOT, GS_SLOT
typedef struct { uint64_t ib; uint32_t itype, first, count; int32_t base_vertex; uint32_t base_instance, pad; } geo_mesh_args;   // dmsh_mesh.py dm_mesh_draw
static struct { int on, miss; __unsafe_unretained id<MTLFunction> fn; __unsafe_unretained NvShader *lean; geo_mesh_args a; } g_geo_mesh;
// The mesh entry for this draw's program, or nil (the compute path draws it)
static id<MTLFunction> geo_mesh_fn(NvProg *pr, const p_draw *dw, const p_draw_idx *di) {
    if (!geo_mesh_on()) return nil;
    int prim = dw ? dw->prim : di->prim, instances = dw ? dw->instances : di->instances, count = dw ? dw->count : di->count;
    NvShader *vs = pr->sh[0];
    if (prim != 4 || count < 3 || instances > 1 || (dw ? dw->ind : di->ind) || (di && (S.restart.on || di->itype < 0 || di->itype > 2)) || S.rast_discard || S.tf_on ||
        !vs || !vs->has_pull || vs->layer_out || vs->discard_fn || !vs->fn || pr->sh[3] || pr->sh[4]) return nil;
    NvShader *lean = geo_lean_pick(pr, NULL, 3, instances); if (!lean) return nil;
    if (!vs->geo_mesh) vs->geo_mesh = [NSMutableDictionary dictionary];
    NSString *k = @(lean->key); id f = vs->geo_mesh[k];
    if (!f) {
        NSString *path = [NSString stringWithFormat:@"%@/%.16s_%.16s_m3.metallib", g_msl_dir, vs->key, lean->key];
        id<MTLLibrary> lib = [NSFileManager.defaultManager fileExistsAtPath:path] ? [g_dev newLibraryWithURL:[NSURL fileURLWithPath:path] error:nil] : nil;
        f = lib ? [lib newFunctionWithName:[vs->fn.name isEqualToString:@"main0_z"] ? @"dm_mesh_z" : @"dm_mesh"] : nil;
        static int n; if (f || n++ < 16) MLOG("geometry mesh: vertex %s kernel %s %s", vs->key, lean->key, f ? "has a mesh library" : "has no mesh library (geo_mesh.tsv): the compute path draws it");
        vs->geo_mesh[k] = f = f ?: NSNull.null;
    }
    return f == NSNull.null ? nil : f;
}
static nvn_hmap g_pso_mesh = {.klen = sizeof(pso_key)}; static os_unfair_lock g_pso_mesh_lock = OS_UNFAIR_LOCK_INIT;
static NSObject *g_pso_mesh_building;
// The draw's mesh pipeline, or nil while it is built in the background (or when it failed)
static id<MTLRenderPipelineState> geo_mesh_pso(NvProg *pr) {
    NvShader *vs = pr->sh[0], *fs = pr->sh[1];
    pso_key k; pso_key_fill(&k, vs, fs, g_pass.cfmt, g_pass.dfmt, g_pass.sfmt, g_pass.samples, 0);
    memset(k.attr, 0, sizeof k.attr); memset(k.stream, 0, sizeof k.stream);   // the pipeline pulls its streams: no vertex layout in it
    k.fsfn = (__bridge void *)g_geo_mesh.fn;
    os_unfair_lock_lock(&g_pso_mesh_lock); id ps = hm_get_id(&g_pso_mesh, &k);
    if (!ps) { if (!g_pso_mesh_building) g_pso_mesh_building = [NSObject new]; hm_put_id(&g_pso_mesh, &k, g_pso_mesh_building); }
    os_unfair_lock_unlock(&g_pso_mesh_lock);
    if (!ps) {
        MTLRenderPipelineDescriptor *d = pso_desc(&k, vs, fs, nil, 0);
        MTLMeshRenderPipelineDescriptor *m = [MTLMeshRenderPipelineDescriptor new];
        m.meshFunction = g_geo_mesh.fn; m.fragmentFunction = d.fragmentFunction;
        m.rasterSampleCount = d.rasterSampleCount; m.alphaToCoverageEnabled = d.alphaToCoverageEnabled;
        for (int i = 0; i < 8; i++) m.colorAttachments[i] = d.colorAttachments[i];
        m.depthAttachmentPixelFormat = d.depthAttachmentPixelFormat; m.stencilAttachmentPixelFormat = d.stencilAttachmentPixelFormat;
        const char *vk = vs->key, *fk = fs ? fs->key : "-", *gk = g_geo_mesh.lean->key; pso_key kk = k;
        dispatch_async(pp_bg_queue(), ^{
            NSError *e = nil; uint64_t t0 = bp_now();
            id<MTLRenderPipelineState> p = [g_dev newRenderPipelineStateWithMeshDescriptor:m options:MTLPipelineOptionNone reflection:nil error:&e];
            MLOG("%sgeometry mesh pipeline: vertex %s fragment %s kernel %s, %.1f ms in the background%s%s", p ? "" : "[RED] ", vk, fk, gk, (bp_now() - t0) / 1e6, p ? "" : ": ", p ? "" : e.localizedDescription.UTF8String);
            os_unfair_lock_lock(&g_pso_mesh_lock); hm_put_id(&g_pso_mesh, &kk, p ?: NSNull.null); os_unfair_lock_unlock(&g_pso_mesh_lock);
        });
        return nil;
    }
    return ps == NSNull.null || ps == g_pso_mesh_building ? nil : ps;
}
// The draw itself (draw() calls it with its pipeline, the vertex and fragment arguments and the pulling layout bound): the kernel's arguments, with vb_info
// m0 = (0, 1 instance) and m1.x = the draw's triangles, and the draw's index source
static void geo_mesh_encode(id<MTLRenderCommandEncoder> en_) {
    id<MTL4RenderCommandEncoder> en = (id<MTL4RenderCommandEncoder>)en_;
    [en setArgumentTable:g_vt atStages:MTLRenderStageMesh];
    uint32_t tris = g_geo_mesh.a.count / 3;
    id<MTLBuffer> b; size_t o = arena_alloc(64, &b); uint8_t *p = (uint8_t *)b.contents + o;
    uint32_t info[8] = {0, 1, 0, 0, tris, 0, 0, 0}; memcpy(p, info, 32); memcpy(p + 32, &g_geo_mesh.a, sizeof g_geo_mesh.a);
    enc_r_vbuf(en, GEO_MESH_DRAW_SLOT, b.gpuAddress + o + 32);
    uint64_t save[5]; memcpy(save, g_geo_bind, sizeof save); memset(g_geo_bind, 0, sizeof g_geo_bind); g_geo_bind[3] = b.gpuAddress + o;
    g_args_dirty[2] = 1; g_args_mesh = GEO_MESH_GS_SLOT; encode_args(en, g_geo_mesh.lean, 2, 0); g_args_mesh = 0; g_args_dirty[2] = 1;
    memcpy(g_geo_bind, save, sizeof save);
    [en drawMeshThreadgroups:MTLSizeMake((tris + GEO_MESH_PRIMS - 1) / GEO_MESH_PRIMS, 1, 1) threadsPerObjectThreadgroup:MTLSizeMake(1, 1, 1) threadsPerMeshThreadgroup:MTLSizeMake(3 * GEO_MESH_PRIMS, 1, 1)];
}
static uint64_t g_geo_mesh_n, g_geo_mesh_wait;   // mesh draws; draws the compute path drew while their mesh pipeline was built
static int geo_mesh_draw(NvProg *pr, const p_draw *dw, const p_draw_idx *di) {
    id<MTLFunction> fn = geo_mesh_fn(pr, dw, di); if (!fn) return 0;
    uint64_t ib = 0; void *mb; uint64_t mo;
    if (di && !mtl_gpu_addr(di->ibuf, 1, &mb, &mo, &ib)) return 0;
    uint32_t bi = (uint32_t)(dw ? dw->base_instance : di->base_instance);
    NvProg *raster = pr->geo_lean_raster;
    if (!raster || raster->sh[0] != pr->sh[0] || raster->sh[1] != pr->sh[1]) { raster = [NvProg new]; raster->sh[0] = pr->sh[0]; raster->sh[1] = pr->sh[1]; pr->geo_lean_raster = raster; }
    g_geo_mesh.on = 1; g_geo_mesh.miss = 0; g_geo_mesh.fn = fn; g_geo_mesh.lean = pr->sh[2]->geo_lean_v[3];
    g_geo_mesh.a = (geo_mesh_args){ib, di ? 1u << di->itype : 0, dw ? (uint32_t)dw->first : 0, (uint32_t)(dw ? dw->count : di->count), di ? di->base_vertex : 0, bi, 0};
    g_geo_program = raster;
    p_draw synthetic = {.prim = 4, .count = 3, .instances = 1, .base_instance = (int32_t)bi};
    draw(&synthetic, NULL);
    g_geo_program = nil; g_geo_mesh.on = 0; args_dirty_all();
    if (g_geo_mesh.miss) { g_geo_mesh_wait++; return 0; }
    swrite_note(pr); udc_note_draw(pr); g_wr_epoch++; g_mdx_epoch++;   // the kernel's storage writes
    if (!g_geo_mesh_n++) MLOG("geometry mesh: first mesh draw, vertex %s kernel %s, %u triangles", pr->sh[0]->key, g_geo_mesh.lean->key, g_geo_mesh.a.count / 3);
    return 1;
}
static void geo_draw_(NvProg *pr, const p_draw *dw0, const p_draw_idx *di0);
static void geo_draw(NvProg *pr, const p_draw *dw0, const p_draw_idx *di0) {
    if (!g_geo_split_depth && geo_mesh_draw(pr, dw0, di0)) return;
    static long left = -1; static uint64_t from; if (left < 0) { const char *e = getenv("DM_NVN_GEO_LEAN_CHECK"); left = e ? atol(e) : 0; from = e && strchr(e, '@') ? strtoull(strchr(e, '@') + 1, NULL, 10) : 0; }
    if (left > 0 && !g_geo_ck_mode && atomic_load(&g_cur_frame) >= from) {
        memset(g_geo_ck, 0, sizeof g_geo_ck);
        g_geo_ck_mode = 1; geo_draw_(pr, dw0, di0);
        g_geo_ck_mode = 2; geo_draw_(pr, dw0, di0);
        g_geo_ck_mode = 0;
        int drained = 0;
        if (g_geo_ck[1].on && g_geo_ck[2].on && geo_sync_cpu(&drained)) { left--; geo_check_compare(); }
        geo_draw_(pr, dw0, di0); return;
    }
    if (g_geo_ready_n && g_exec_cur && !g_geo_split_depth)   // prepared ahead of the pass: only the raster is owed
        for (int i = 0; i < g_geo_ready_n; i++) if (g_geo_ready[i].op == g_exec_cur && g_geo_ready[i].sub == g_sub_serial) {
            geo_ready g = g_geo_ready[i]; g_geo_ready[i] = g_geo_ready[--g_geo_ready_n];
            PD("G %llu geometry draw prepared ahead of the pass: raster only", g_stat_pass + !g_enc);
            geo_raster(pr, &g); return;
        }
    if (g_exec_next && !g_geo_split_depth && !g_geo_prep_out && geo_hoist_on()) {
        geo_ready self = {0}; g_geo_prep_out = &self; geo_draw_(pr, dw0, di0); g_geo_prep_out = NULL;
        if (self.op) { geo_hoist_scan(); geo_flush(); geo_raster(pr, &self); return; }
    }
    geo_draw_(pr, dw0, di0);
}
static void geo_draw_(NvProg *pr, const p_draw *dw0, const p_draw_idx *di0) {
    NvShader *vs = pr->sh[0], *base = pr->sh[2];
    if (!vs || !vs->geo_fn || !base || !base->geo_vertices || pr->sh[3] || pr->sh[4]) {
        MLOG("[RED] geometry draw cannot capture this pipeline (vertex %s, geometry %s, tessellation %d, feedback %d)", vs ? vs->key : "-", base ? base->key : "-", !!(pr->sh[3] || pr->sh[4]), S.tf_on); return;
    }
    p_draw dw = dw0 ? *dw0 : (p_draw){0}; p_draw_idx di = di0 ? *di0 : (p_draw_idx){0};
    if (di0 && (di.itype < 0 || di.itype > 2)) { MLOG("[RED] geometry index type %d is invalid", di.itype); return; }
    uint64_t ind = dw0 ? dw.ind : di.ind;
    if (g_geo_prep_out && (ind || !geo_lean_pick(pr, NULL, nvn_geo_topology(dw0 ? dw.prim : di.prim), dw0 ? dw.instances : di.instances) || geo_mesh_fn(pr, dw0, di0))) return;   // decided before any wait; a mesh draw needs no preparation
    if (defer_has_active()) defer_fallback();
    lc_land(); rp_flush(); rp_end(); ubd_land();
    int drained = g_geo_split_depth != 0;
    // Decode indirect arguments only after their CPU snapshot is safe. They
    // select the index range checked below; no GPU work is encoded between the
    // two checks, so a successful drain covers both and the converted streams.
    if ((geo_conversion_hazard(vs) || (ind && geo_cpu_hazard(ind, dw0 ? 16 : 20))) && !geo_sync_cpu(&drained)) return;
    if (ind) {
        uint32_t a[5] = {0}; size_t n = dw0 ? 16 : 20;
        if (!nvn_host_readable(ind, n)) { MLOG("[RED] geometry indirect arguments are not readable"); return; }
        memcpy(a, (void *)(uintptr_t)ind, n);
        if (!a[0] || !a[1]) return;
        if (a[0] > INT32_MAX || a[1] > INT32_MAX) { MLOG("[RED] geometry indirect draw exceeds the host count range"); return; }
        if (dw0) { dw.count = a[0]; dw.instances = a[1]; dw.first = a[2]; dw.base_instance = a[3]; dw.ind = dw.mdi_cnt = 0; }
        else { di.count = a[0]; di.instances = a[1]; di.ibuf += (uint64_t)a[2] << di.itype; di.base_vertex = (int32_t)a[3]; di.base_instance = a[4]; di.ind = di.mdi_cnt = 0; }
    }
    int prim = dw0 ? dw.prim : di.prim, topology = nvn_geo_topology(prim);
    int count = dw0 ? dw.count : di.count, instances = dw0 ? dw.instances : di.instances;
    uint32_t first_instance = (uint32_t)(dw0 ? dw.base_instance : di.base_instance);
    g_draw_binst = first_instance;
    if (count <= 0 || instances < 0) return;
    if (!instances) instances = 1;
    if (topology < 0 || count > (1 << 22)) { MLOG("[RED] geometry primitive %d or count %d exceeds the assembly path", prim, count); return; }
    if (di0 && geo_cpu_hazard(di.ibuf, (uint64_t)count << di.itype) && !geo_sync_cpu(&drained)) return;
    NSString *key = [[NSString stringWithUTF8String:base->key] substringToIndex:16];
    NvShader *gs = shader_get([key stringByAppendingFormat:@"_bg%d", topology], 2);
    NvShader *reader = shader_get([key stringByAppendingString:@"_bgv"], 0);
    if (!gs || !reader) return;
    NvShader *lean = g_geo_ck_mode == 1 ? NULL : geo_lean_pick(pr, gs, topology, instances);   // check mode 1: the full kernel
    if ((g_geo_ck_mode == 2 || g_geo_prep_out) && !lean) return;
    NvShader *kernel = lean ? lean : gs;
    id<MTLRenderPipelineState> capture = geo_capture_pso(vs); if (!capture) return;
    id<MTLComputePipelineState> __strong *lean_k = g_geo_lean_k, *pack = g_geo_pack;
    if (!pack[2]) { NSError *e = nil; id<MTLLibrary> lib = [g_dev newLibraryWithSource:@(k_nvn_geo_pack_msl) options:nil error:&e];
        const char *fn[3] = {"geo_pack_count", "geo_pack_scan", "geo_pack_write"};
        for (int i = 0; i < 3 && lib; i++) if (!(pack[i] = [g_dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@(fn[i])] error:&e])) break;
        if (!pack[2]) { pack[0] = pack[1] = nil; MLOG("[RED] geometry index compaction: %s", e.localizedDescription.UTF8String); return; }
        if (lib) for (int i = 0; i < 3; i++) lean_k[i] = [g_dev newComputePipelineStateWithFunction:[lib newFunctionWithName:i == 2 ? @"geo_lean_tri" : i ? @"geo_lean_index" : @"geo_lean_cull"] error:&e];
        if (!lean_k[0] || !lean_k[1] || !lean_k[2]) { lean_k[0] = lean_k[1] = lean_k[2] = nil; MLOG("[RED] lean geometry index kernels: %s", e.localizedDescription.UTF8String); } }
    if (!lean_k[1]) lean = NULL, kernel = gs;
    if ((g_geo_ck_mode == 2 || g_geo_prep_out) && !lean) return;
    if (!gs->geo_pso && !lean) { NSError *e = nil; gs->geo_pso = [g_dev newComputePipelineStateWithFunction:gs->fn error:&e];
        if (!gs->geo_pso) { MLOG("[RED] geometry compute %s: %s", gs->key, e.localizedDescription.UTF8String); return; } }
    if (kernel->geo_pso.maxTotalThreadsPerThreadgroup < 32) { MLOG("[RED] geometry kernel cannot run a 32-thread group"); return; }

    uint32_t *order = NULL;
    uint32_t lo = UINT32_MAX, hi = 0; size_t nu = 0;
    int restart = di0 && S.restart.on;
    int isz = di0 ? 1 << di.itype : 0;
    uint32_t restart_index = S.restart.index & (isz == 1 ? UINT8_MAX : isz == 2 ? UINT16_MAX : UINT32_MAX);
    if (di0 && (di.itype < 0 || di.itype > 2 || !nvn_host_readable(di.ibuf, (uint64_t)count * isz))) {
        MLOG("[RED] geometry index buffer is not readable"); return;
    }
    // Typed loops: a memcpy call per index took most of the time of geometry-heavy frames (NieR:Automata, 80 such draws a frame)
    typedef uint16_t u16_any __attribute__((aligned(1))); typedef uint32_t u32_any __attribute__((aligned(1)));
    const void *ib = di0 ? (const void *)(uintptr_t)di.ibuf : NULL;
    #define GEO_INDEX(i) (isz == 1 ? ((const uint8_t *)ib)[i] : isz == 2 ? ((const u16_any *)ib)[i] : ((const u32_any *)ib)[i])
    // A lean list draw without restart (nearly all of NieR:Automata's) needs no index array of its own: one pass finds the range, and the
    // primitive map is the indices themselves, widened straight into the scratch (below). Its whole range is captured when the range has no
    // more vertices than the draw has indices; a sparser range needs the unique-vertex list, from the array.
    int fast = lean && di0 && !restart && (prim == 0 || prim == 1 || prim == 4) && instances == 1 && !vs->discard_fn;
    if (fast) {
        if (isz == 2) for (int i = 0; i < count; i++) { uint32_t v = ((const u16_any *)ib)[i]; lo = MIN(lo, v); hi = MAX(hi, v); }
        else for (int i = 0; i < count; i++) { uint32_t v = GEO_INDEX(i); lo = MIN(lo, v); hi = MAX(hi, v); }
        nu = (size_t)count;
        if ((uint64_t)hi - lo + 1 > (uint64_t)count) fast = 0;
    }
    if (!fast) {
        order = malloc((size_t)count * 4);
        if (!order) { MLOG("[RED] geometry index allocation failed"); return; }
        if (!di0) for (int i = 0; i < count; i++) order[i] = (uint32_t)i;
        else if (isz == 1) for (int i = 0; i < count; i++) order[i] = ((const uint8_t *)ib)[i];
        else if (isz == 2) for (int i = 0; i < count; i++) order[i] = ((const u16_any *)ib)[i];
        else for (int i = 0; i < count; i++) order[i] = ((const u32_any *)ib)[i];
        lo = UINT32_MAX; hi = 0; nu = 0;
        if (!restart) { for (int i = 0; i < count; i++) { lo = MIN(lo, order[i]); hi = MAX(hi, order[i]); } nu = (size_t)count; }
        else for (int i = 0; i < count; i++) { uint32_t v = order[i]; if (v == restart_index) continue; lo = MIN(lo, v); hi = MAX(hi, v); nu++; }
    }
    if (!nu) { free(order); return; }
    int64_t first = di0 ? (int64_t)lo + di.base_vertex : dw.first;
    uint64_t span = (uint64_t)hi - lo + 1;
    if (first < 0 || first + span > (uint64_t)UINT32_MAX + 1) { free(order); MLOG("[RED] geometry vertex range is outside 32-bit vertex IDs"); return; }
    // Normalize only after assembly, so a restart value cannot collide with a
    // normalized vertex number. Indexed captures use a sorted unique index list.
    size_t words = fast ? (size_t)count / (size_t)nvn_geo_width(topology) * (size_t)nvn_geo_width(topology) : nvn_geo_assemble(prim, order, count, restart, restart_index, NULL, 0);
    int width = nvn_geo_width(topology);
    uint64_t np = words == SIZE_MAX ? 0 : words / width;
    if (!np) { free(order); return; }
    uint64_t groups = np * (uint64_t)instances * kernel->geo_invocations;
    if (groups > UINT32_MAX || span * (uint64_t)instances > GEO_CAPTURE_VERTICES ||
        (uint64_t)first_instance + instances > (uint64_t)UINT32_MAX + 1) {
        free(order);
        if (g_geo_prep_out) return;   // drawn in runs: not prepared ahead
        if (!geo_split(pr, dw0 ? &dw : NULL, dw0 ? NULL : &di, count, instances, first_instance)) MLOG("[RED] geometry draw exceeds the capture or invocation bound"); return;
    }
    uint32_t record = lean ? lean->geo_words : GEO_CAPTURE_WORDS;   // words per captured vertex
    uint64_t input_size = span * instances * MAX(record, 1u) * 4;
    uint64_t output_size = groups * kernel->geo_vertices * kernel->geo_stride * 4;
    uint64_t raw_size = groups * kernel->geo_index_stride * 4;
    uint64_t packed_size = groups * kernel->geo_vertices * (kernel->geo_output == 7 ? 3 : kernel->geo_output == 6 ? 2 : 1) * 4;
    uint64_t total = input_size + output_size + raw_size + packed_size + groups * 4 + words * 4 + nu * 4 + 4096 + 1024;
    if (groups > UINT32_MAX || total > (256ull << 20) || np > UINT32_MAX) {
        free(order);
        if (g_geo_prep_out) return;   // drawn in runs: not prepared ahead
        if (!geo_split(pr, dw0 ? &dw : NULL, dw0 ? NULL : &di, count, instances, first_instance)) MLOG("[RED] geometry draw needs %llu MiB of scratch, above the 256 MiB bound", (unsigned long long)(total >> 20)); return;
    }
    id<MTLBuffer> scratch; size_t off = arena_alloc((size_t)total, &scratch);
    uint8_t *cpu = (uint8_t *)scratch.contents + off; uint64_t gpu = scratch.gpuAddress + off;
    size_t used = 0;
    #define GEO_ALLOC(N, G, P) do { used = (used + 255) & ~(size_t)255; (G) = gpu + used; (P) = cpu + used; used += (size_t)(N); } while (0)
    uint64_t input, output, raw, packed, cnt, map, capture_ix, info, capture_info, args, job;
    void *pinput, *poutput, *praw, *ppacked, *pcnt, *pmap, *pcapture_ix, *pinfo, *pcapture_info, *pargs, *pjob;
    GEO_ALLOC(input_size, input, pinput); GEO_ALLOC(output_size, output, poutput); GEO_ALLOC(raw_size, raw, praw); GEO_ALLOC(packed_size, packed, ppacked); GEO_ALLOC(groups * 4, cnt, pcnt);
    GEO_ALLOC(words * 4, map, pmap); GEO_ALLOC(nu * 4, capture_ix, pcapture_ix); GEO_ALLOC(32, info, pinfo); GEO_ALLOC(sizeof(nvn_geo_capture), capture_info, pcapture_info);
    GEO_ALLOC(20, args, pargs); GEO_ALLOC(sizeof(nvn_geo_pack_job), job, pjob);
    uint64_t ljob; void *pljob; GEO_ALLOC(sizeof(nvn_geo_lean_job), ljob, pljob);
    #undef GEO_ALLOC
    (void)pcnt;
    if (used > total) MDIE("geometry scratch size accounting");
    // A raw map (the fast path): the capture's records start lo records before the input, so record i is index i's, and the vertex index is
    // map + base vertex; else the map is normalized to lo and the vertex index is map + first
    uint64_t lead = (uint64_t)lo * MAX(record, 1u) * 4;
    int raw_map = fast && lead <= input;
    if (fast) { uint32_t *m = pmap, sub = raw_map ? 0 : lo;
        if (isz == 2) for (size_t i = 0; i < words; i++) m[i] = (uint32_t)((const u16_any *)ib)[i] - sub;
        else for (size_t i = 0; i < words; i++) m[i] = (uint32_t)GEO_INDEX(i) - sub; }
    else {
        nvn_geo_assemble(prim, order, count, restart, restart_index, pmap, words);
        for (size_t i = 0; i < words; i++) ((uint32_t *)pmap)[i] -= lo;
    }
    #undef GEO_INDEX
    // When the index range has no more vertices than the draw has indices (a mesh's own range, the usual case), its whole range is captured by a
    // plain draw instead of building the unique-vertex list; not for a vertex shader with storage writes, which would run on unreferenced vertices
    int whole = di0 && !vs->discard_fn && span <= (uint64_t)count;
    size_t unique_count = di0 && !whole ? geo_capture_unique(order, count, restart, restart_index, lo, span, pcapture_ix, nu) : 0;
    free(order);
    if (unique_count == SIZE_MAX) { MLOG("[RED] geometry capture indices exceed their range or allocation"); return; }
    uint32_t vi[8] = {(uint32_t)span, (uint32_t)instances, 0, 0, (uint32_t)np, 0, 0, 0}; memcpy(pinfo, vi, 32);
    nvn_geo_capture gc = raw_map ? (nvn_geo_capture){input - lead, 0, {(uint32_t)di.base_vertex, hi + 1, first_instance, 1}, MAX(record, 1u), {0}}
                                 : (nvn_geo_capture){input, 0, {(uint32_t)first, (uint32_t)span, first_instance, (uint32_t)instances}, MAX(record, 1u), {0}};
    for (int k = 0; k < GEO_CAPTURE_WORDS; k++) gc.slot[k] = lean ? lean->geo_slot[k] : (int16_t)k;
    memcpy(pcapture_info, &gc, sizeof gc);
    memset(pargs, 0, 20);
    nvn_geo_pack_job pj = {raw, packed, args, (uint32_t)groups, kernel->geo_index_stride, kernel->geo_output, (uint32_t)(packed_size / 4), cnt};
    memcpy(pjob, &pj, sizeof pj);
    nvn_geo_lean_job lj = {raw, output, map, packed, args, (uint32_t)groups, kernel->geo_index_stride, raw_map ? (uint32_t)di.base_vertex : (uint32_t)first, first_instance, (uint32_t)(packed_size / 4), (uint32_t)first};
    memcpy(pljob, &lj, sizeof lj);
    vbuf_prepare();
    p_draw vertex_range = {.first = (int32_t)first, .count = (int32_t)span, .instances = instances, .base_instance = (int32_t)first_instance};
    if (g_geo_njobs > GEO_READY_MAX) geo_flush();
    void *state = malloc(sizeof S);
    if (!state) { MLOG("[RED] geometry job allocation failed"); return; }
    memcpy(state, &S, sizeof S);
    int tri = lean && topology == 3;
    g_geo_jobs[g_geo_njobs++] = (geo_job){state, capture, vs, kernel, lean != NULL, tri, di0 && !whole, instances, record, first_instance, (uint32_t)unique_count,
        di0 ? di.base_vertex : 0, whole ? (int32_t)first : dw0 ? dw.first : 0, whole ? (int32_t)span : dw0 ? dw.count : 0, capture_info, capture_ix, nu * 4, {raw_map ? input - lead : input, output, raw, info, map}, np, groups, job, ljob, packed_size, vertex_range};
    if (!g_geo_prep_out) geo_flush();   // drawn now: its pre-work goes out at once; a run prepared ahead goes out together (geo_draw)
    swrite_note(pr); udc_note_draw(pr); g_wr_epoch++; g_mdx_epoch++;
    if (S.tf_on) tf_geometry(reader, packed, args, gs->geo_output == 1 ? 1 : gs->geo_output == 6 ? 2 : 3, first_instance);
    if (g_geo_prep_out) {
        *g_geo_prep_out = (geo_ready){g_exec_cur, g_sub_serial, vs, packed, packed_size, args, vertex_range, first_instance, topology, S.rast_discard, tri ? (uint32_t)(groups * 3) : 0};
        return;
    }
    if (g_geo_ck_mode) {
        g_geo_ck[g_geo_ck_mode] = (typeof(g_geo_ck[0])){1, topology == 0 ? 1 : topology == 1 ? 2 : 3, kernel->geo_stride, (uint32_t)first, praw, poutput, ppacked, pargs, pmap, pinput, tri ? (uint32_t)(groups * 3) : 0};
        return;
    }
    if (S.rast_discard) return;
    if (lean) {   // the program's own stages through the kept primitives' vertex indices
        geo_ready g = {g_exec_cur, g_sub_serial, vs, packed, packed_size, args, vertex_range, first_instance, topology, 0, tri ? (uint32_t)(groups * 3) : 0};
        geo_raster(pr, &g);
        static int once; if (!once++) MLOG("geometry compute, lean: vertex %s, geometry %s reads %u capture words, %llu input primitives; the vertex shader draws", vs->key, lean->key, record, (unsigned long long)np);
        return;
    }
    NvProg *raster = pr->geo_raster;
    if (!raster || raster->sh[0] != reader || raster->sh[1] != pr->sh[1]) {
        raster = [NvProg new]; raster->sh[0] = reader; raster->sh[1] = pr->sh[1]; pr->geo_raster = raster;
    }
    g_geo_program = raster; g_geo_indices = packed; g_geo_index_bytes = packed_size; g_geo_args = args;
    p_draw synthetic = {.prim = gs->geo_output == 1 ? 0 : gs->geo_output == 6 ? 1 : 4, .count = 3, .instances = 1, .base_instance = (int32_t)first_instance};
    draw(&synthetic, NULL);
    g_geo_program = nil; args_dirty_all();
    static int once; if (!once++) MLOG("geometry compute: vertex %s, geometry %s, %llu input primitives, %d instances", vs->key, gs->key, (unsigned long long)np, instances);
}
