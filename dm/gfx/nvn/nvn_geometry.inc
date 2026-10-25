// General geometry replay: native VS capture -> GS compute -> ordered index
// compaction -> ordinary rendering. Included after the resource/pipeline helpers.
static NvProg *g_geo_program;
static uint64_t g_geo_indices, g_geo_index_bytes, g_geo_args;
static void draw(const p_draw *dw, const p_draw_idx *di);
static void lc_land(void);

// One geometry draw's compute part, prepared on the CPU and encoded before its raster draw.
// A run of geometry draws with only state between them (NieR:Automata's depth prepass:
// one per mesh, about 70 a frame) is encoded as one capture pass and one compute pass by
// the run's first draw; each later draw then only rasterizes, in the open render pass.
// Done one by one, each draw ended the pass and ran four encoders: 20 ms of GPU a frame.
typedef struct {
    const uint8_t *op; uint64_t sub;   // the draw's op in this submission (the run's later draws), NULL for the first
    NvProg *pr; NvShader *vs, *gs, *reader; id<MTLRenderPipelineState> capture;
    p_draw dw; int indexed, instances, emits; uint32_t first_instance, base_vertex; int64_t first;
    uint64_t span, np, groups, nu, unique_count, input, output, raw, info, map, packed, packed_size, cnt, args, job, capture_ix, capture_info;
} geo_job;
#define GEO_RUN 64
static geo_job g_geo_run[GEO_RUN]; static int g_geo_nrun;
static id<MTLComputePipelineState> g_geo_pack[3];   // geo_pack_count, geo_pack_scan, geo_pack_write (nvn_geometry.h)

static int geo_cmp_index(const void *a, const void *b) {
    uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
    return (x > y) - (x < y);
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
static int geo_capturable(NvProg *pr) {
    NvShader *vs = pr->sh[0], *base = pr->sh[2];
    return vs && vs->geo_fn && base && base->geo_vertices && !pr->sh[3] && !pr->sh[4] && !S.tf_on;
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
// The draw's CPU part from the current state: primitive assembly, scratch, the kernels' jobs.
// → 1 prepared, 0 nothing to draw, -1 its data needs a GPU wait first (may_wait 0: a later draw of a run)
static int geo_prep(NvProg *pr, const p_draw *dw0, const p_draw_idx *di0, geo_job *j, int may_wait) {
    NvShader *vs = pr->sh[0], *base = pr->sh[2];
    p_draw dw = dw0 ? *dw0 : (p_draw){0}; p_draw_idx di = di0 ? *di0 : (p_draw_idx){0};
    if (di0 && (di.itype < 0 || di.itype > 2)) { MLOG("[RED] geometry index type %d is invalid", di.itype); return 0; }
    uint64_t ind = dw0 ? dw.ind : di.ind;
    // Indirect records and index data written on the GPU in this submission need a
    // coherent CPU snapshot for primitive assembly; indices no queued write covers are
    // read as they are, as in ix_range. A wait at every indexed draw held NieR:Automata's
    // submit for about 190 ms a frame (an index-build geometry draw per mesh).
    int wait = ind != 0 || (di0 && di.count > 0 && pend_hits(di.ibuf, (uint64_t)di.count << di.itype));
    for (int i = 0; i < 16; i++) if (S.vbuf[i].size && pend_hits(S.vbuf[i].address, S.vbuf[i].size)) wait = 1;
    for (int i = 0; i < vs->rf.nv; i++) if (vconv_wanted(&vs->rf, i, NULL, 0)) wait = 1;
    if (g_geo_split_depth) wait = 0;   // a part of a split draw: the whole draw waited already
    if (wait && !may_wait) return -1;
    if (wait) { mdi_wait(); if (g_mdi_ev.signaledValue < g_mdi_v) return 0; memset(g_vconv, 0, sizeof g_vconv); }
    if (ind) {
        uint32_t a[5] = {0}; size_t n = dw0 ? 16 : 20;
        if (!nvn_host_readable(ind, n)) { MLOG("[RED] geometry indirect arguments are not readable"); return 0; }
        memcpy(a, (void *)(uintptr_t)ind, n);
        if (!a[0] || !a[1]) return 0;
        if (a[0] > INT32_MAX || a[1] > INT32_MAX) { MLOG("[RED] geometry indirect draw exceeds the host count range"); return 0; }
        if (dw0) { dw.count = a[0]; dw.instances = a[1]; dw.first = a[2]; dw.base_instance = a[3]; dw.ind = dw.mdi_cnt = 0; }
        else { di.count = a[0]; di.instances = a[1]; di.ibuf += (uint64_t)a[2] << di.itype; di.base_vertex = (int32_t)a[3]; di.base_instance = a[4]; di.ind = di.mdi_cnt = 0; }
    }
    int prim = dw0 ? dw.prim : di.prim, topology = nvn_geo_topology(prim);
    int count = dw0 ? dw.count : di.count, instances = dw0 ? dw.instances : di.instances;
    uint32_t first_instance = (uint32_t)(dw0 ? dw.base_instance : di.base_instance);
    if (count <= 0 || instances < 0) return 0;
    if (!instances) instances = 1;
    if (topology < 0 || count > (1 << 22)) { MLOG("[RED] geometry primitive %d or count %d exceeds the assembly path", prim, count); return 0; }
    NSString *key = [[NSString stringWithUTF8String:base->key] substringToIndex:16];
    NvShader *gs = shader_get([key stringByAppendingFormat:@"_bg%d", topology], 2);
    NvShader *reader = shader_get([key stringByAppendingString:@"_bgv"], 0);
    if (!gs || !reader) return 0;
    id<MTLRenderPipelineState> capture = geo_capture_pso(vs); if (!capture) return 0;
    if (!gs->geo_pso) { NSError *e = nil; gs->geo_pso = [g_dev newComputePipelineStateWithFunction:gs->fn error:&e];
        if (!gs->geo_pso) { MLOG("[RED] geometry compute %s: %s", gs->key, e.localizedDescription.UTF8String); return 0; } }
    if (gs->geo_pso.maxTotalThreadsPerThreadgroup < 32) { MLOG("[RED] geometry kernel cannot run a 32-thread group"); return 0; }
    if (!g_geo_pack[2]) { NSError *e = nil; id<MTLLibrary> lib = [g_dev newLibraryWithSource:@(k_nvn_geo_pack_msl) options:nil error:&e];
        const char *fn[3] = {"geo_pack_count", "geo_pack_scan", "geo_pack_write"};
        for (int i = 0; i < 3 && lib; i++) if (!(g_geo_pack[i] = [g_dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@(fn[i])] error:&e])) break;
        if (!g_geo_pack[2]) { g_geo_pack[0] = g_geo_pack[1] = nil; MLOG("[RED] geometry index compaction: %s", e.localizedDescription.UTF8String); return 0; } }

    uint32_t *order = malloc((size_t)count * 4), *unique = malloc((size_t)count * 4);
    if (!order || !unique) { free(order); free(unique); MLOG("[RED] geometry index allocation failed"); return 0; }
    uint32_t lo = UINT32_MAX, hi = 0; size_t nu = 0;
    int restart = di0 && S.restart.on;
    int isz = di0 ? 1 << di.itype : 0;
    uint32_t restart_index = S.restart.index & (isz == 1 ? UINT8_MAX : isz == 2 ? UINT16_MAX : UINT32_MAX);
    if (di0 && !nvn_host_readable(di.ibuf, (uint64_t)count * isz)) {
        free(order); free(unique); MLOG("[RED] geometry index buffer is not readable"); return 0;
    }
    for (int i = 0; i < count; i++) {
        uint32_t v = (uint32_t)i;
        if (di0) { v = 0; memcpy(&v, (const uint8_t *)(uintptr_t)di.ibuf + (size_t)i * isz, isz); }
        order[i] = v;
        if (restart && v == restart_index) continue;
        lo = MIN(lo, v); hi = MAX(hi, v); unique[nu++] = v;
    }
    if (!nu) { free(order); free(unique); return 0; }
    int64_t first = di0 ? (int64_t)lo + di.base_vertex : dw.first;
    uint64_t span = (uint64_t)hi - lo + 1;
    if (first < 0 || first + span > (uint64_t)UINT32_MAX + 1) { free(order); free(unique); MLOG("[RED] geometry vertex range is outside 32-bit vertex IDs"); return 0; }
    // Normalize only after assembly, so a restart value cannot collide with a
    // normalized vertex number. Indexed captures use a sorted unique index list.
    size_t words = nvn_geo_assemble(prim, order, count, restart, restart_index, NULL, 0);
    int width = nvn_geo_width(topology);
    uint64_t np = words == SIZE_MAX ? 0 : words / width;
    if (!np) { free(order); free(unique); return 0; }
    uint64_t groups = np * (uint64_t)instances * gs->geo_invocations;
    if (groups > UINT32_MAX || span * (uint64_t)instances > (256ull << 20) / (144 * 4) ||
        (uint64_t)first_instance + instances > (uint64_t)UINT32_MAX + 1) {
        free(order); free(unique); if (!may_wait) return -1;
        if (!geo_split(pr, dw0 ? &dw : NULL, dw0 ? NULL : &di, count, instances, first_instance)) MLOG("[RED] geometry draw exceeds the capture or invocation bound"); return 0;
    }
    uint64_t input_size = span * instances * 144 * 4;
    uint64_t output_size = groups * gs->geo_vertices * gs->geo_stride * 4;
    uint64_t raw_size = groups * gs->geo_index_stride * 4;
    uint64_t packed_size = groups * gs->geo_vertices * (gs->geo_output == 7 ? 3 : gs->geo_output == 6 ? 2 : 1) * 4;
    uint64_t total = input_size + output_size + raw_size + packed_size + groups * 4 + words * 4 + nu * 4 + 4096;
    if (groups > UINT32_MAX || total > (256ull << 20) || np > UINT32_MAX) {
        free(order); free(unique); if (!may_wait) return -1;
        if (!geo_split(pr, dw0 ? &dw : NULL, dw0 ? NULL : &di, count, instances, first_instance)) MLOG("[RED] geometry draw needs %llu MiB of scratch, above the 256 MiB bound", (unsigned long long)(total >> 20)); return 0;
    }
    id<MTLBuffer> scratch; size_t off = arena_alloc((size_t)total, &scratch);
    uint8_t *cpu = (uint8_t *)scratch.contents + off; uint64_t gpu = scratch.gpuAddress + off;
    size_t used = 0;
    #define GEO_ALLOC(N, G, P) do { used = (used + 255) & ~(size_t)255; (G) = gpu + used; (P) = cpu + used; used += (size_t)(N); } while (0)
    void *pinput, *poutput, *praw, *ppacked, *pcnt, *pmap, *pcapture_ix, *pinfo, *pcapture_info, *pargs, *pjob;
    GEO_ALLOC(input_size, j->input, pinput); GEO_ALLOC(output_size, j->output, poutput); GEO_ALLOC(raw_size, j->raw, praw); GEO_ALLOC(packed_size, j->packed, ppacked); GEO_ALLOC(groups * 4, j->cnt, pcnt);
    GEO_ALLOC(words * 4, j->map, pmap); GEO_ALLOC(nu * 4, j->capture_ix, pcapture_ix); GEO_ALLOC(32, j->info, pinfo); GEO_ALLOC(32, j->capture_info, pcapture_info);
    GEO_ALLOC(20, j->args, pargs); GEO_ALLOC(sizeof(nvn_geo_pack_job), j->job, pjob);
    #undef GEO_ALLOC
    (void)pinput; (void)poutput; (void)praw; (void)ppacked; (void)pcnt;
    if (used > total) MDIE("geometry scratch size accounting");
    nvn_geo_assemble(prim, order, count, restart, restart_index, pmap, words);
    for (size_t i = 0; i < words; i++) ((uint32_t *)pmap)[i] -= lo;
    qsort(unique, nu, 4, geo_cmp_index);
    size_t unique_count = 0;
    for (size_t i = 0; i < nu; i++) if (!i || unique[i] != unique[i - 1]) ((uint32_t *)pcapture_ix)[unique_count++] = unique[i];
    free(order); free(unique);
    uint32_t vi[8] = {(uint32_t)span, (uint32_t)instances, 0, 0, (uint32_t)np, 0, 0, 0}; memcpy(pinfo, vi, 32);
    uint64_t ci[2] = {j->input, 0}; memcpy(pcapture_info, ci, 16);
    uint32_t cc[4] = {(uint32_t)first, (uint32_t)span, first_instance, (uint32_t)instances}; memcpy((uint8_t *)pcapture_info + 16, cc, 16);
    memset(pargs, 0, 20);
    nvn_geo_pack_job pj = {j->raw, j->packed, j->args, (uint32_t)groups, gs->geo_index_stride, gs->geo_output, (uint32_t)(packed_size / 4), j->cnt};
    memcpy(pjob, &pj, sizeof pj);
    // A kernel without the vertex output helper emits nothing: no compaction and no raster pass
    int emits = 0;
    for (int i = 0; i < gs->rf.nb; i++) if (gs->rf.b[i].kind == NVN_RB_GEOMETRY && gs->rf.b[i].reg == 1) emits = 1;
    j->pr = pr; j->vs = vs; j->gs = gs; j->reader = reader; j->capture = capture;
    j->dw = dw; j->indexed = di0 != NULL; j->instances = instances; j->emits = emits; j->first_instance = first_instance; j->base_vertex = (uint32_t)di.base_vertex;
    j->first = first; j->span = span; j->np = np; j->groups = groups; j->nu = nu; j->unique_count = unique_count; j->packed_size = packed_size;
    return 1;
}
// Encoders take the current state: the vertex streams and uniforms of the draw's own place in the replay
static void geo_capture(id<MTL4RenderCommandEncoder> re, const geo_job *j) {
    enc_r_pso(re, j->capture); bind_vbufs(re);
    p_draw vertex_range = {.first = (int32_t)j->first, .count = (int32_t)j->span, .instances = j->instances, .base_instance = (int32_t)j->first_instance};
    vconv_bind(re, j->vs, 1, &vertex_range, NULL);
    g_draw_binst = j->first_instance;
    args_dirty_all(); encode_args(re, j->vs, 0, 1); enc_r_vbuf(re, 13, j->capture_info);
    if (j->indexed) enc_r_draw_indexed(re, MTLPrimitiveTypePoint, j->unique_count, MTLIndexTypeUInt32, j->capture_ix, j->nu * 4, j->instances, (int32_t)j->base_vertex, j->first_instance);
    else enc_r_draw_instanced(re, MTLPrimitiveTypePoint, j->dw.first, j->dw.count, j->instances, j->first_instance);
}
static void geo_bind(const geo_job *j) { g_geo_bind[0] = j->input; g_geo_bind[1] = j->output; g_geo_bind[2] = j->raw; g_geo_bind[3] = j->info; g_geo_bind[4] = j->map; }
static void geo_dispatch(id<MTL4ComputeCommandEncoder> ce, const geo_job *j) {
    geo_bind(j);
    g_draw_binst = j->first_instance;
    args_dirty_all(); encode_args(ce, j->gs, 2, 0); enc_c_pso(ce, j->gs->geo_pso);
    enc_c_dispatch(ce, MTLSizeMake((j->np + 31) / 32, j->instances, j->gs->geo_invocations), MTLSizeMake(32, 1, 1));
}
// The ops a run continues across: state only (replayed by state_op), and query reports and
// event signals, which write memory no geometry draw's compute part reads
static int geo_run_passes(uint32_t op, const void *a) {
    if (op == OP_REPORT_COUNTER || op == OP_SIGNAL_EVENT || op == OP_RESET_COUNTER) return 1;
    return state_op(op, a);
}
// Walks the run's later draws from the op after the current draw, replaying the state between
// them; prep: prepare and capture each (→ how many joined), else dispatch the n prepared
static int geo_run_walk(id<MTL4CommandEncoder> en, int prep, int n) {
    static typeof(S) Ssave; memcpy(&Ssave, &S, sizeof S);
    p_scissor sc = R.sc; int has_sc = R.has_sc, pso_dirty = g_pso_dirty, dss_dirty = g_dss_dirty;
    fz_cursor cur = {.p = g_exec_next}; int k = 1;
    while (k < (prep ? GEO_RUN : n)) {
        const uint8_t *q; op_hdr h; const void *a;
        if (!fz_next(&cur, &q, &h, &a)) break;
        if (h.op != OP_DRAW && h.op != OP_DRAW_IDX) { if (geo_run_passes(h.op, a)) continue; break; }
        if (!prep) { if (q != g_geo_run[k].op) break; geo_dispatch((id<MTL4ComputeCommandEncoder>)en, &g_geo_run[k]); k++; continue; }
        p_draw tdw; p_draw_idx tdi; const p_draw *d = h.op == OP_DRAW ? draw_rec(a, h.size, &tdw) : NULL; const p_draw_idx *x = h.op == OP_DRAW_IDX ? draw_idx_rec(a, h.size, &tdi) : NULL;
        __unsafe_unretained NvProg *p2 = S.prog ? prog_get(S.prog) : nil;
        if (!p2 || !p2->sh[2] || !p2->sh[2]->geo_vertices || !geo_capturable(p2) || S.rast_discard || (d && d->mdi_cnt) || (x && x->mdi_cnt)) break;
        geo_job *j = &g_geo_run[k];
        if (geo_prep(p2, d, x, j, 0) <= 0) break;
        j->op = q; j->sub = g_sub_serial;
        geo_capture((id<MTL4RenderCommandEncoder>)en, j); k++;
    }
    memcpy(&S, &Ssave, sizeof S); R.sc = sc; R.has_sc = has_sc; args_dirty_all(); g_pso_dirty = pso_dirty; g_dss_dirty = dss_dirty; prog_resolve();
    return k;
}
static void geo_raster(const geo_job *j) {
    NvProg *pr = j->pr;
    swrite_note(pr); udc_note_draw(pr);
    if (S.rast_discard || !j->emits) return;
    NvProg *raster = pr->geo_raster;
    if (!raster || raster->sh[0] != j->reader || raster->sh[1] != pr->sh[1]) {
        raster = [NvProg new]; raster->sh[0] = j->reader; raster->sh[1] = pr->sh[1]; pr->geo_raster = raster;
    }
    geo_bind(j);   // the reader takes the draw's own geometry output (a run dispatched later draws since)
    g_geo_program = raster; g_geo_indices = j->packed; g_geo_index_bytes = j->packed_size; g_geo_args = j->args;
    p_draw synthetic = {.prim = j->gs->geo_output == 1 ? 0 : j->gs->geo_output == 6 ? 1 : 4, .count = 3, .instances = 1, .base_instance = (int32_t)j->first_instance};
    g_draw_binst = j->first_instance;
    draw(&synthetic, NULL);
    g_geo_program = nil; args_dirty_all();
}
static void geo_draw(NvProg *pr, const p_draw *dw0, const p_draw_idx *di0) {
    // A later draw of a run: its compute part was encoded by the run's first draw
    if (g_exec_cur) for (int i = 1; i < g_geo_nrun; i++) if (g_geo_run[i].op == g_exec_cur && g_geo_run[i].sub == g_sub_serial && g_geo_run[i].pr == pr) {
        g_geo_run[i].op = NULL; geo_raster(&g_geo_run[i]); return; }
    g_geo_nrun = 0;
    if (!geo_capturable(pr)) {
        NvShader *vs = pr->sh[0], *base = pr->sh[2];
        MLOG("[RED] geometry draw cannot capture this pipeline (vertex %s, geometry %s, tessellation %d, feedback %d)", vs ? vs->key : "-", base ? base->key : "-", !!(pr->sh[3] || pr->sh[4]), S.tf_on); return;
    }
    if (defer_has_active()) defer_fallback();
    lc_land(); rp_flush(); rp_end(); ubd_land();
    geo_job *j0 = &g_geo_run[0];
    if (geo_prep(pr, dw0, di0, j0, 1) <= 0) return;
    j0->op = NULL;
    if (g_blit) { enc_b_end(g_blit); g_blit = nil; }
    MTL4RenderPassDescriptor *rd = [MTL4RenderPassDescriptor new]; rd.renderTargetWidth = rd.renderTargetHeight = rd.defaultRasterSampleCount = 1;
    id<MTL4RenderCommandEncoder> re = enc_r_begin(rd);
    geo_capture(re, j0);
    int n = g_exec_next && !g_geo_split_depth ? geo_run_walk(re, 1, 0) : 1;
    enc_r_end(re);
    id<MTL4ComputeCommandEncoder> ce = enc_c_begin(NULL);   // the run's kernels are independent: no barrier between them
    geo_dispatch(ce, j0);
    if (n > 1) geo_run_walk(ce, 0, n);
    int packs = 0; for (int i = 0; i < n; i++) packs |= g_geo_run[i].emits;
    for (int st = 0; packs && st < 3; st++) {   // count per group, offsets (one thread), write per group: each stage waits for the one before, for the whole run
        [ce barrierAfterEncoderStages:MTLStageDispatch beforeEncoderStages:MTLStageDispatch visibilityOptions:MTL4VisibilityOptionDevice];
        enc_c_pso(ce, g_geo_pack[st]);
        for (int i = 0; i < n; i++) if (g_geo_run[i].emits) { enc_c_args(ce, g_geo_run[i].job, 0);
            enc_c_dispatch(ce, st == 1 ? MTLSizeMake(1, 1, 1) : MTLSizeMake((g_geo_run[i].groups + 63) / 64, 1, 1), MTLSizeMake(st == 1 ? 1 : 64, 1, 1)); }
    }
    enc_c_end(ce);
    g_geo_nrun = n;
    args_dirty_all(); g_enc_pso = nil; g_enc_dss = nil; g_ed.ok = 0;
    g_wr_epoch++; g_mdx_epoch++;
    static int once; if (!once++) MLOG("geometry compute: vertex %s, geometry %s, %llu input primitives, %d instances", j0->vs->key, j0->gs->key, (unsigned long long)j0->np, j0->instances);
    static uint64_t runs, draws; runs++; draws += n;
    if (runs == 100) MLOG("geometry compute: %llu draws in the first 100 runs", (unsigned long long)draws);
    geo_raster(j0);
}
