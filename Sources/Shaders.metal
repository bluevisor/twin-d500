// Twin D500: a ray-marched Mandelbulb rendered by two GPUs at once.
// Each GPU gets a band of rows. The display GPU writes straight into the
// drawable; the headless GPU writes packed BGRA pixels into system memory
// that the display GPU then copies into the lower band.

#include <metal_stdlib>
using namespace metal;

struct Uniforms {
    float4 resTime;   // x = width, y = height, z = time, w = bulb power
    float4 camPos;    // xyz = eye
    float4 camTarget; // xyz = look-at point
    int4   rows;      // x = first row, y = row count, z = samples per axis, w = max march steps
    int4   flags;     // x = tint bands, y = gpu index, z = shadows + reflections, w = buffer stride (px)
};

static float3 rotY(float3 p, float a) {
    float c = cos(a), s = sin(a);
    return float3(c * p.x + s * p.z, p.y, -s * p.x + c * p.z);
}

static float3 sunDir() { return normalize(float3(0.55, 0.72, -0.42)); }

// Mandelbulb distance estimate. trap collects orbit-trap values for coloring.
static float bulb(float3 p, float power, thread float4 &trap) {
    float3 z = p;
    float dr = 1.0;
    float r = length(z);
    trap = float4(abs(z), dot(z, z));
    for (int i = 0; i < 10; i++) {
        r = length(z);
        if (r > 2.0) break;
        float theta = acos(clamp(z.y / r, -1.0, 1.0)) * power;
        float phi = atan2(z.z, z.x) * power;
        float rp = pow(r, power - 1.0);
        dr = rp * power * dr + 1.0;
        z = (rp * r) * float3(sin(theta) * cos(phi), cos(theta), sin(theta) * sin(phi)) + p;
        trap = min(trap, float4(abs(z), dot(z, z)));
    }
    return 0.5 * log(r) * r / dr;
}

// Scene: the bulb plus a floor. mat 0 = bulb, 1 = floor.
static float map(float3 p, float time, float power, thread float4 &trap, thread int &mat) {
    float3 q = rotY(p, time * 0.12);
    float qq = dot(q, q);
    float d = qq > 1.8 ? sqrt(qq) - 1.25 : bulb(q, power, trap);
    if (qq > 1.8) trap = float4(1.0);
    float fl = p.y + 1.2;
    mat = 0;
    if (fl < d) { mat = 1; d = fl; }
    return d;
}

static float sceneDist(float3 p, constant Uniforms &u) {
    float4 tr; int m;
    return map(p, u.resTime.z, u.resTime.w, tr, m);
}

static float march(float3 ro, float3 rd, constant Uniforms &u, int maxSteps,
                   thread float4 &trap, thread int &mat, thread float &glow) {
    float t = 0.0;
    glow = 0.0;
    for (int i = 0; i < maxSteps; i++) {
        float d = map(ro + rd * t, u.resTime.z, u.resTime.w, trap, mat);
        if (mat == 0) glow += 0.012 / (0.012 + d * d * 60.0);
        if (d < 0.00035 * t + 0.0002) return t;
        t += d * 0.92;
        if (t > 25.0) break;
    }
    return -1.0;
}

static float3 calcNormal(float3 p, float t, constant Uniforms &u) {
    float e = max(0.0004, 0.0006 * t);
    const float2 k = float2(1.0, -1.0);
    return normalize(k.xyy * sceneDist(p + k.xyy * e, u) +
                     k.yyx * sceneDist(p + k.yyx * e, u) +
                     k.yxy * sceneDist(p + k.yxy * e, u) +
                     k.xxx * sceneDist(p + k.xxx * e, u));
}

static float softShadow(float3 ro, float3 rd, constant Uniforms &u) {
    float res = 1.0, t = 0.012;
    for (int i = 0; i < 56; i++) {
        float h = sceneDist(ro + rd * t, u);
        res = min(res, 9.0 * h / t);
        t += clamp(h, 0.006, 0.25);
        if (res < 0.002 || t > 6.0) break;
    }
    return clamp(res, 0.0, 1.0);
}

static float ambientOcclusion(float3 p, float3 n, constant Uniforms &u) {
    float occ = 0.0, sca = 1.0;
    for (int i = 0; i < 5; i++) {
        float h = 0.012 + 0.07 * float(i);
        occ += (h - sceneDist(p + n * h, u)) * sca;
        sca *= 0.8;
    }
    return clamp(1.0 - 2.2 * occ, 0.0, 1.0);
}

static float3 palette(float t) {
    return 0.5 + 0.5 * cos(6.28318 * (float3(1.0, 0.8, 0.6) * t + float3(0.02, 0.18, 0.32)));
}

static float3 skyColor(float3 rd) {
    float3 col = mix(float3(0.42, 0.30, 0.34), float3(0.03, 0.04, 0.10), clamp(rd.y * 1.6 + 0.15, 0.0, 1.0));
    float sd = max(dot(rd, sunDir()), 0.0);
    col += float3(1.0, 0.62, 0.32) * (pow(sd, 6.0) * 0.30 + pow(sd, 300.0) * 3.0);
    return col;
}

// Lighting for one surface hit (no reflections, so it can be reused for the floor mirror).
static float3 shadeSurface(float3 p, float3 rd, float t, float4 trap, int mat,
                           constant Uniforms &u, bool fancy) {
    float3 n = calcNormal(p, t, u);
    float3 sun = sunDir();
    float3 base;
    if (mat == 0) {
        float k = sqrt(trap.w) * 1.25 + trap.y * 0.7;
        base = palette(k + u.resTime.z * 0.015);
        base = mix(base, float3(0.95, 0.9, 0.82), clamp(trap.x * 1.6, 0.0, 1.0) * 0.15);
        base *= base;
    } else {
        float2 g = p.xz * 1.2;
        float checker = fmod(floor(g.x) + floor(g.y), 2.0) == 0.0 ? 1.0 : 0.0;
        base = mix(float3(0.02, 0.022, 0.03), float3(0.05, 0.052, 0.06), checker);
    }
    float sh = fancy ? softShadow(p + n * 0.002, sun, u) : 1.0;
    float occ = fancy ? ambientOcclusion(p, n, u) : 0.7;
    float diff = max(dot(n, sun), 0.0) * sh;
    float amb = 0.5 + 0.5 * n.y;
    float spec = pow(max(dot(reflect(rd, n), sun), 0.0), mat == 0 ? 24.0 : 64.0) * sh;
    float fre = pow(clamp(1.0 + dot(n, rd), 0.0, 1.0), 5.0);
    float3 col = base * (diff * float3(2.6, 2.2, 1.8) + amb * occ * float3(0.30, 0.40, 0.58));
    col += spec * float3(1.0, 0.88, 0.7) * (mat == 0 ? 0.7 : 0.35);
    col += fre * occ * skyColor(reflect(rd, n)) * (mat == 0 ? 0.35 : 0.15);
    return col;
}

static float3 shadePixel(float2 frag, constant Uniforms &u) {
    float2 res = u.resTime.xy;
    float2 uv = (2.0 * frag - res) / res.y;
    uv.y = -uv.y;
    float3 ro = u.camPos.xyz;
    float3 ww = normalize(u.camTarget.xyz - ro);
    float3 uu = normalize(cross(ww, float3(0.0, 1.0, 0.0)));
    float3 vv = cross(uu, ww);
    float3 rd = normalize(uv.x * uu + uv.y * vv + 1.9 * ww);
    bool fancy = u.flags.z != 0;
    int steps = u.rows.w;

    float4 trap; int mat; float glow;
    float t = march(ro, rd, u, steps, trap, mat, glow);
    float3 col;
    if (t < 0.0) {
        col = skyColor(rd);
    } else {
        float3 p = ro + rd * t;
        col = shadeSurface(p, rd, t, trap, mat, u, fancy);
        if (mat == 1 && fancy) {
            // Glossy floor: one mirror bounce toward the bulb.
            float3 n = float3(0.0, 1.0, 0.0);
            float3 rr = reflect(rd, n);
            float3 rp = p + n * 0.003;
            float4 trap2; int mat2; float glow2;
            float t2 = march(rp, rr, u, steps / 2, trap2, mat2, glow2);
            float3 refl = (t2 > 0.0 && mat2 == 0)
                ? shadeSurface(rp + rr * t2, rr, t2, trap2, mat2, u, false)
                : skyColor(rr);
            float fre = 0.18 + 0.5 * pow(clamp(1.0 + dot(rd, n), 0.0, 1.0), 4.0);
            col = mix(col, refl, fre);
        }
        float fogT = 1.0 - exp(-0.0045 * t * t);
        col = mix(col, skyColor(rd) * 0.9, fogT);
    }
    col += float3(1.0, 0.45, 0.18) * glow * 0.016;
    return col;
}

static float3 finish(float3 col, float2 frag, float2 res) {
    col = (col * (2.51 * col + 0.03)) / (col * (2.43 * col + 0.59) + 0.14); // ACES fit
    col = pow(clamp(col, 0.0, 1.0), float3(1.0 / 2.2));
    float2 q = frag / res;
    col *= 0.55 + 0.45 * pow(16.0 * q.x * q.y * (1.0 - q.x) * (1.0 - q.y), 0.18);
    return col;
}

static float3 renderPixel(uint2 gid, constant Uniforms &u, int y) {
    int spp = max(u.rows.z, 1);
    float3 acc = float3(0.0);
    for (int sy = 0; sy < spp; sy++)
        for (int sx = 0; sx < spp; sx++) {
            float2 off = (float2(sx, sy) + 0.5) / float(spp);
            acc += shadePixel(float2(gid.x, y) + off, u);
        }
    float2 frag = float2(gid.x, y) + 0.5;
    float3 col = finish(acc / float(spp * spp), frag, u.resTime.xy);
    if (u.flags.x != 0) {
        float3 tint = u.flags.y == 0 ? float3(0.25, 0.6, 1.0) : float3(1.0, 0.55, 0.15);
        col = mix(col, tint, 0.16);
        if (y == u.rows.x && y != 0) col = float3(1.0);
    }
    return col;
}

// Display GPU: writes its band straight into the target texture.
kernel void renderTex(texture2d<float, access::write> out [[texture(0)]],
                      constant Uniforms &u [[buffer(0)]],
                      uint2 gid [[thread_position_in_grid]]) {
    if (int(gid.x) >= int(u.resTime.x) || int(gid.y) >= u.rows.y) return;
    int y = int(gid.y) + u.rows.x;
    out.write(float4(renderPixel(gid, u, y), 1.0), uint2(gid.x, y));
}

// Headless GPU: writes its band as packed BGRA8 into a shared system-memory buffer.
kernel void renderBuf(device uchar4 *out [[buffer(1)]],
                      constant Uniforms &u [[buffer(0)]],
                      uint2 gid [[thread_position_in_grid]]) {
    if (int(gid.x) >= int(u.resTime.x) || int(gid.y) >= u.rows.y) return;
    int y = int(gid.y) + u.rows.x;
    float3 c = renderPixel(gid, u, y);
    uchar3 b = uchar3(clamp(c, 0.0, 1.0) * 255.0 + 0.5);
    out[gid.y * uint(u.flags.w) + gid.x] = uchar4(b.z, b.y, b.x, 255);
}
