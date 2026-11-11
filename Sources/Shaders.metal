// Twin D500 shaders.
//
// Pixel scenes are split by rows: the display GPU writes its band straight into
// the frame, the headless GPU writes packed BGRA pixels into system memory that
// the display GPU copies into the lower band. The galaxy scene splits the
// N-body simulation by bodies instead.

#include <metal_stdlib>
using namespace metal;

struct Uniforms {
    float4 resTime;   // x = width, y = height, z = time, w = bulb power (morph mode)
    float4 camPos;    // xyz = orbit camera eye (Mandelbulb, Nebula)
    float4 camTarget; // xyz = orbit camera look-at point
    int4   rows;      // x = first row, y = row count, z = samples per axis, w = max march steps
    int4   flags;     // x = tint bands, y = gpu index, z = shadows + reflections, w = buffer stride (px)
    int4   scene;     // x = scene (0 bulb, 1 canyon, 2 sponge, 3 nebula, 4 gallery), y = morph, z = quality 0-2
};

// Compiled into a separate pipeline per scene, so each kernel only carries one scene's code.
// (One kernel holding every scene needs the register budget of the heaviest one, which
// cuts how many pixels a GCN compute unit can keep in flight: measured ~25% slower.)
constant int  kScene [[function_constant(0)]];
constant bool kMorph [[function_constant(1)]];

// ============================================================== helpers

static float3 rotY(float3 p, float a) {
    float c = cos(a), s = sin(a);
    return float3(c * p.x + s * p.z, p.y, -s * p.x + c * p.z);
}

static float hash12(float2 p) {
    float3 p3 = fract(float3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

static float hash13(float3 p3) {
    p3 = fract(p3 * 0.1031);
    p3 += dot(p3, p3.zyx + 31.32);
    return fract((p3.x + p3.y) * p3.z);
}

static float noise2(float2 x) {
    float2 i = floor(x), f = fract(x);
    float2 u = f * f * (3.0 - 2.0 * f);
    float a = hash12(i), b = hash12(i + float2(1, 0));
    float c = hash12(i + float2(0, 1)), d = hash12(i + float2(1, 1));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

static float noise3(float3 x) {
    float3 i = floor(x), f = fract(x);
    float3 u = f * f * (3.0 - 2.0 * f);
    float n000 = hash13(i),                    n100 = hash13(i + float3(1, 0, 0));
    float n010 = hash13(i + float3(0, 1, 0)),  n110 = hash13(i + float3(1, 1, 0));
    float n001 = hash13(i + float3(0, 0, 1)),  n101 = hash13(i + float3(1, 0, 1));
    float n011 = hash13(i + float3(0, 1, 1)),  n111 = hash13(i + float3(1, 1, 1));
    return mix(mix(mix(n000, n100, u.x), mix(n010, n110, u.x), u.y),
               mix(mix(n001, n101, u.x), mix(n011, n111, u.x), u.y), u.z);
}

static float3 cameraRay(float2 frag, float2 res, float3 ro, float3 ta, float zoom, float roll) {
    float2 uv = (2.0 * frag - res) / res.y;
    uv.y = -uv.y;
    float cr = cos(roll), sr = sin(roll);
    uv = float2(cr * uv.x - sr * uv.y, sr * uv.x + cr * uv.y);
    float3 ww = normalize(ta - ro);
    float3 uu = normalize(cross(ww, float3(0.0, 1.0, 0.0)));
    float3 vv = cross(uu, ww);
    return normalize(uv.x * uu + uv.y * vv + zoom * ww);
}

// ============================================================== scene 0: Mandelbulb

static float3 sunDir() { return normalize(float3(0.55, 0.72, -0.42)); }

// Trig form: any power, used when the shape morphs.
static float bulbTrig(float3 p, float power, thread float4 &trap) {
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

// Polynomial form: power 8 only, no transcendentals in the loop (GCN runs those at quarter rate).
static float bulbPoly(float3 p, thread float4 &trap) {
    float3 w = p;
    float m = dot(w, w);
    trap = float4(abs(w), m);
    float dz = 1.0;
    for (int i = 0; i < 5; i++) {
        float m2 = m * m;
        dz = 8.0 * sqrt(m2 * m2 * m2 * m) * dz + 1.0;
        float x = w.x, x2 = x * x, x4 = x2 * x2;
        float y = w.y, y2 = y * y, y4 = y2 * y2;
        float z = w.z, z2 = z * z, z4 = z2 * z2;
        float k3 = x2 + z2;
        float k2 = rsqrt(k3 * k3 * k3 * k3 * k3 * k3 * k3);
        float k1 = x4 + y4 + z4 - 6.0 * y2 * z2 - 6.0 * x2 * y2 + 2.0 * z2 * x2;
        float k4 = x2 - y2 + z2;
        w.x = p.x + 64.0 * x * y * z * (x2 - z2) * k4 * (x4 - 6.0 * x2 * z2 + z4) * k1 * k2;
        w.y = p.y + -16.0 * y2 * k3 * k4 * k4 + k1 * k1;
        w.z = p.z + -8.0 * y * k4 * (x4 * x4 - 28.0 * x4 * x2 * z2 + 70.0 * x4 * z4 - 28.0 * x2 * z2 * z4 + z4 * z4) * k1 * k2;
        m = dot(w, w);
        trap = min(trap, float4(abs(w), m));
        if (m > 256.0) break;
    }
    return 0.25 * log(m) * sqrt(m) / dz;
}

// mat 0 = bulb, 1 = floor.
static float bulbMap(float3 p, constant Uniforms &u, thread float4 &trap, thread int &mat) {
    float3 q = rotY(p, u.resTime.z * 0.12);
    float qq = dot(q, q);
    float d;
    if (qq > 1.8) { d = sqrt(qq) - 1.25; trap = float4(1.0); }
    else d = kMorph ? bulbTrig(q, u.resTime.w, trap) : bulbPoly(q, trap);
    float fl = p.y + 1.2;
    mat = 0;
    if (fl < d) { mat = 1; d = fl; }
    return d;
}

static float bulbDist(float3 p, constant Uniforms &u) {
    float4 tr; int m;
    return bulbMap(p, u, tr, m);
}

static float bulbMarch(float3 ro, float3 rd, constant Uniforms &u, int maxSteps,
                       thread float4 &trap, thread int &mat, thread float &glow) {
    float t = 0.0;
    glow = 0.0;
    for (int i = 0; i < maxSteps; i++) {
        float d = bulbMap(ro + rd * t, u, trap, mat);
        if (mat == 0) glow += 0.012 / (0.012 + d * d * 60.0);
        if (d < 0.00035 * t + 0.0002) return t;
        t += d * 0.92;
        if (t > 25.0) break;
    }
    return -1.0;
}

static float3 bulbNormal(float3 p, float t, constant Uniforms &u) {
    float e = max(0.0004, 0.0006 * t);
    const float2 k = float2(1.0, -1.0);
    return normalize(k.xyy * bulbDist(p + k.xyy * e, u) + k.yyx * bulbDist(p + k.yyx * e, u) +
                     k.yxy * bulbDist(p + k.yxy * e, u) + k.xxx * bulbDist(p + k.xxx * e, u));
}

static float bulbShadow(float3 ro, float3 rd, constant Uniforms &u) {
    float res = 1.0, t = 0.012;
    for (int i = 0; i < 56; i++) {
        float h = bulbDist(ro + rd * t, u);
        res = min(res, 9.0 * h / t);
        t += clamp(h, 0.006, 0.25);
        if (res < 0.002 || t > 6.0) break;
    }
    return clamp(res, 0.0, 1.0);
}

static float bulbAO(float3 p, float3 n, constant Uniforms &u) {
    float occ = 0.0, sca = 1.0;
    for (int i = 0; i < 5; i++) {
        float h = 0.012 + 0.07 * float(i);
        occ += (h - bulbDist(p + n * h, u)) * sca;
        sca *= 0.8;
    }
    return clamp(1.0 - 2.2 * occ, 0.0, 1.0);
}

static float3 bulbPalette(float t) {
    return 0.5 + 0.5 * cos(6.28318 * (float3(1.0, 0.8, 0.6) * t + float3(0.02, 0.18, 0.32)));
}

static float3 bulbSky(float3 rd) {
    float3 col = mix(float3(0.42, 0.30, 0.34), float3(0.03, 0.04, 0.10), clamp(rd.y * 1.6 + 0.15, 0.0, 1.0));
    float sd = max(dot(rd, sunDir()), 0.0);
    col += float3(1.0, 0.62, 0.32) * (pow(sd, 6.0) * 0.30 + pow(sd, 300.0) * 3.0);
    return col;
}

static float3 bulbSurface(float3 p, float3 rd, float t, float4 trap, int mat, constant Uniforms &u, bool fancy) {
    float3 n = bulbNormal(p, t, u);
    float3 sun = sunDir();
    float3 base;
    if (mat == 0) {
        float k = sqrt(trap.w) * 1.25 + trap.y * 0.7;
        base = bulbPalette(k + u.resTime.z * 0.015);
        base = mix(base, float3(0.95, 0.9, 0.82), clamp(trap.x * 1.6, 0.0, 1.0) * 0.15);
        base *= base;
    } else {
        float2 g = p.xz * 1.2;
        float checker = fmod(floor(g.x) + floor(g.y), 2.0) == 0.0 ? 1.0 : 0.0;
        base = mix(float3(0.02, 0.022, 0.03), float3(0.05, 0.052, 0.06), checker);
    }
    float sh = fancy ? bulbShadow(p + n * 0.002, sun, u) : 1.0;
    float occ = fancy ? bulbAO(p, n, u) : 0.7;
    float diff = max(dot(n, sun), 0.0) * sh;
    float amb = 0.5 + 0.5 * n.y;
    float spec = pow(max(dot(reflect(rd, n), sun), 0.0), mat == 0 ? 24.0 : 64.0) * sh;
    float fre = pow(clamp(1.0 + dot(n, rd), 0.0, 1.0), 5.0);
    float3 col = base * (diff * float3(2.6, 2.2, 1.8) + amb * occ * float3(0.30, 0.40, 0.58));
    col += spec * float3(1.0, 0.88, 0.7) * (mat == 0 ? 0.7 : 0.35);
    col += fre * occ * bulbSky(reflect(rd, n)) * (mat == 0 ? 0.35 : 0.15);
    return col;
}

static float3 shadeBulb(float2 frag, float2 res, constant Uniforms &u) {
    float3 ro = u.camPos.xyz;
    float3 rd = cameraRay(frag, res, ro, u.camTarget.xyz, 1.9, 0.0);
    bool fancy = u.flags.z != 0;
    int steps = u.rows.w;
    float4 trap; int mat; float glow;
    float t = bulbMarch(ro, rd, u, steps, trap, mat, glow);
    float3 col;
    if (t < 0.0) {
        col = bulbSky(rd);
    } else {
        float3 p = ro + rd * t;
        col = bulbSurface(p, rd, t, trap, mat, u, fancy);
        if (mat == 1 && fancy) {
            float3 n = float3(0.0, 1.0, 0.0);
            float3 rr = reflect(rd, n);
            float3 rp = p + n * 0.003;
            float4 trap2; int mat2; float glow2;
            float t2 = bulbMarch(rp, rr, u, steps / 2, trap2, mat2, glow2);
            float3 refl = (t2 > 0.0 && mat2 == 0) ? bulbSurface(rp + rr * t2, rr, t2, trap2, mat2, u, false) : bulbSky(rr);
            float fre = 0.18 + 0.5 * pow(clamp(1.0 + dot(rd, n), 0.0, 1.0), 4.0);
            col = mix(col, refl, fre);
        }
        col = mix(col, bulbSky(rd) * 0.9, 1.0 - exp(-0.0045 * t * t));
    }
    col += float3(1.0, 0.45, 0.18) * glow * 0.016;
    return col;
}

// ============================================================== scene 1: canyon flyover

static float3 canyonSun() { return normalize(float3(-0.62, 0.30, 0.72)); }

// Ridged multifractal: each octave's ridges get sharper where the previous one was high.
static float canyonHeight(float2 p, int octaves) {
    float2 q = p * 0.045;
    float h = 0.0, a = 0.6, w = 1.0;
    for (int i = 0; i < octaves; i++) {
        float n = noise2(q);
        n = 1.0 - abs(n * 2.0 - 1.0);
        n = n * n * w;
        w = clamp(n * 1.8, 0.0, 1.0);
        h += a * n;
        a *= 0.5;
        q = float2(1.6 * q.x - 1.2 * q.y, 1.2 * q.x + 1.6 * q.y) + float2(3.1, 1.7);
    }
    return h * 26.0 - 6.0;
}

static float3 canyonPath(float t) {
    return float3(30.0 * sin(t * 0.035) + 12.0 * sin(t * 0.09), 0.0, t * 5.0);
}

static float canyonMarch(float3 ro, float3 rd, int steps, float tmax) {
    float t = 0.5;
    for (int i = 0; i < steps; i++) {
        float3 p = ro + rd * t;
        float h = p.y - canyonHeight(p.xz, t < 80.0 ? 8 : 5);
        if (h < 0.0015 * t) return t;
        t += 0.4 * h;
        if (t > tmax) return -1.0;
    }
    return t;
}

static float3 canyonNormal(float3 p, float t) {
    float e = 0.01 + 0.0015 * t;
    float hx = canyonHeight(p.xz + float2(e, 0.0), 10) - canyonHeight(p.xz - float2(e, 0.0), 10);
    float hz = canyonHeight(p.xz + float2(0.0, e), 10) - canyonHeight(p.xz - float2(0.0, e), 10);
    return normalize(float3(-hx, 2.0 * e, -hz));
}

static float canyonShadow(float3 ro, float3 rd) {
    float res = 1.0, t = 0.5;
    for (int i = 0; i < 48; i++) {
        float3 p = ro + rd * t;
        float h = p.y - canyonHeight(p.xz, 5);
        res = min(res, 12.0 * h / t);
        t += clamp(h * 0.6, 0.3, 3.0);
        if (res < 0.01 || t > 80.0) break;
    }
    return clamp(res, 0.0, 1.0);
}

static float3 canyonSky(float3 ro, float3 rd, float time) {
    float3 sun = canyonSun();
    float y = max(rd.y, 0.0);
    float3 col = mix(float3(0.80, 0.62, 0.52), float3(0.22, 0.38, 0.68), pow(y, 0.45));
    float sd = max(dot(rd, sun), 0.0);
    col += float3(1.0, 0.65, 0.35) * (0.25 * pow(sd, 6.0) + 0.6 * pow(sd, 64.0));
    col += float3(1.0, 0.9, 0.7) * 8.0 * pow(sd, 1500.0);
    if (rd.y > 0.0) {
        float2 cp = (ro.xz + rd.xz * (250.0 - ro.y) / rd.y) * 0.0035 + float2(time * 0.01, 0.0);
        float c = 0.0, a = 0.5;
        for (int i = 0; i < 5; i++) { c += a * noise2(cp); cp = cp * 2.1 + 1.3; a *= 0.5; }
        float cov = smoothstep(0.48, 0.8, c) * smoothstep(0.0, 0.15, rd.y);
        float3 cloud = mix(float3(0.95, 0.85, 0.8), float3(1.0, 0.8, 0.6), pow(sd, 4.0));
        col = mix(col, cloud, cov * 0.8);
    }
    return col;
}

static float3 canyonSurface(float3 p, float3 n, bool fancy) {
    float3 sun = canyonSun();
    float r = noise2(p.xz * 0.35);
    float strata = 0.5 + 0.5 * sin(p.y * 1.7 + r * 3.0 + noise2(p.xz * 0.05) * 4.0);
    float3 rock = mix(float3(0.42, 0.19, 0.10), float3(0.58, 0.31, 0.16), 0.3 + 0.4 * strata);
    rock = mix(rock, float3(0.30, 0.22, 0.18), smoothstep(0.55, 0.9, r) * 0.6);
    float3 alb = mix(rock, float3(0.70, 0.50, 0.32), smoothstep(0.80, 0.95, n.y));
    alb = mix(alb, float3(0.92, 0.94, 0.98), smoothstep(13.0, 16.0, p.y + r * 3.0) * smoothstep(0.55, 0.8, n.y));
    alb *= alb;
    float sh = fancy ? canyonShadow(p + n * 0.1, sun) : 1.0;
    float dif = clamp(dot(n, sun), 0.0, 1.0) * sh;
    float sky = clamp(0.5 + 0.5 * n.y, 0.0, 1.0);
    float bounce = clamp(dot(n, normalize(float3(-sun.x, 0.0, -sun.z))), 0.0, 1.0) * clamp(1.0 - n.y, 0.0, 1.0);
    return alb * (dif * float3(4.5, 3.4, 2.4) + sky * float3(0.30, 0.42, 0.62) + bounce * float3(0.35, 0.18, 0.08));
}

static float3 shadeCanyon(float2 frag, float2 res, constant Uniforms &u) {
    float time = u.resTime.z;
    float3 ro = canyonPath(time);
    float ground = max(max(canyonHeight(ro.xz, 5), canyonHeight(canyonPath(time + 1.5).xz, 5)),
                       canyonHeight(canyonPath(time + 3.0).xz, 5));
    ro.y = ground + 5.0;
    float3 ta = canyonPath(time + 4.0);
    ta.y = ro.y - 2.0;
    float3 rd = cameraRay(frag, res, ro, ta, 1.6, 0.08 * sin(time * 0.2));
    int q = u.scene.z;
    float3 sky = canyonSky(ro, rd, time);
    float t = canyonMarch(ro, rd, q == 0 ? 90 : (q == 1 ? 150 : 200), 400.0);
    if (t < 0.0) return sky;
    float3 p = ro + rd * t;
    float3 col = canyonSurface(p, canyonNormal(p, t), u.flags.z != 0);
    float3 fogCol = mix(float3(0.62, 0.55, 0.52), float3(1.0, 0.7, 0.45), pow(max(dot(rd, canyonSun()), 0.0), 6.0));
    return mix(col, fogCol, 1.0 - exp(-0.00006 * t * t));
}

// ============================================================== scene 2: Menger tunnel

static float3 mod3(float3 x, float y) { return x - y * floor(x / y); }

// Infinite lattice of Menger sponges; positive = empty space. trap = carve level that won.
static float spongeDE(float3 p, thread float &trap) {
    float d = -1e5;
    float s = 1.0;
    trap = 0.0;
    for (int i = 0; i < 5; i++) {
        float3 a = mod3(p * s, 2.0) - 1.0;
        s *= 3.0;
        float3 r = abs(1.0 - 3.0 * abs(a));
        float da = max(r.x, r.y), db = max(r.y, r.z), dc = max(r.z, r.x);
        float c = (min(da, min(db, dc)) - 1.0) / s;
        if (c > d) { d = c; trap = float(i); }
    }
    return d;
}

static float spongeDist(float3 p) { float tr; return spongeDE(p, tr); }

static float spongeShadow(float3 ro, float3 rd, float maxt) {
    float res = 1.0, t = 0.004;
    for (int i = 0; i < 32; i++) {
        float h = spongeDist(ro + rd * t);
        res = min(res, 8.0 * h / t);
        t += clamp(h, 0.003, 0.15);
        if (res < 0.005 || t > maxt) break;
    }
    return clamp(res, 0.0, 1.0);
}

static float3 shadeSponge(float2 frag, float2 res, constant Uniforms &u) {
    float time = u.resTime.z;
    float z = time * 0.55;
    float3 ro = float3(0.06 * sin(time * 0.31), 0.06 * cos(time * 0.27), z);
    float3 ta = ro + float3(0.25 * sin(time * 0.11), 0.18 * sin(time * 0.07), 1.0);
    float3 rd = cameraRay(frag, res, ro, ta, 1.3, time * 0.07);
    int steps = u.scene.z == 0 ? 90 : (u.scene.z == 1 ? 140 : 180);
    bool fancy = u.flags.z != 0;
    float3 lp = float3(0.0, 0.0, z + 1.6 + 0.6 * sin(time * 0.5));

    float t = 0.0, trap = 0.0;
    bool hit = false;
    int i = 0;
    for (; i < steps; i++) {
        float d = spongeDE(ro + rd * t, trap);
        if (d < 0.0004 * t + 0.00015) { hit = true; break; }
        t += d;
        if (t > 30.0) break;
    }
    float3 fogCol = float3(0.015, 0.02, 0.04);
    float3 col = fogCol;
    if (hit) {
        float3 p = ro + rd * t;
        float e = 0.0005 + 0.0005 * t;
        const float2 k = float2(1.0, -1.0);
        float3 n = normalize(k.xyy * spongeDist(p + k.xyy * e) + k.yyx * spongeDist(p + k.yyx * e) +
                             k.yxy * spongeDist(p + k.yxy * e) + k.xxx * spongeDist(p + k.xxx * e));
        float3 ld = lp - p;
        float ll = length(ld);
        ld /= ll;
        float atten = 1.6 / (1.0 + ll * ll * 1.5);
        float occ = 1.0 - float(i) / float(steps);
        occ *= occ;
        float sh = fancy ? spongeShadow(p + n * 0.001, ld, ll) : 1.0;
        float3 base = 0.5 + 0.5 * cos(6.28318 * (trap * 0.18 + float3(0.0, 0.33, 0.67)) + p.z * 0.15);
        base = mix(float3(0.55, 0.55, 0.6), base, 0.55);
        float dif = max(dot(n, ld), 0.0) * sh;
        float spec = pow(max(dot(reflect(rd, n), ld), 0.0), 32.0) * sh;
        col = base * (dif * atten * float3(1.0, 0.75, 0.5) * 3.0 + occ * 0.12 * float3(0.3, 0.5, 1.0));
        col += spec * atten * 1.5;
        col += float3(0.2, 0.9, 1.0) * smoothstep(3.5, 4.0, trap) * 0.5 * occ;
        col = mix(col, fogCol, 1.0 - exp(-0.05 * t * t));
    }
    // Visible light orb (closest approach of the ray to the light).
    float3 w = lp - ro;
    float tp = dot(w, rd);
    if (tp > 0.0 && (!hit || tp < t)) {
        float dl = length(w - rd * tp);
        col += float3(1.0, 0.7, 0.4) * 0.004 / (dl * dl + 0.0012);
    }
    return col;
}

// ============================================================== scene 3: nebula

static float nebulaDensity(float3 p, float time) {
    float r = length(p);
    if (r < 0.35 || r > 3.9) return 0.0;
    float3 q = p * 1.15 + float3(0.0, time * 0.03, 0.0);
    q += (noise3(q * 0.35 + 3.7) - 0.5) * 3.0;   // domain warp: swirls instead of round blobs
    float f = 0.0, a = 0.55;
    for (int i = 0; i < 5; i++) { f += a * noise3(q); q = q * 2.03 + float3(1.7, 9.2, 3.3); a *= 0.5; }
    return max(f * 3.6 - 1.85, 0.0) * 2.2 * smoothstep(3.8, 2.2, r) * smoothstep(0.4, 1.3, r);
}

static float3 starField(float3 rd) {
    float3 col = float3(0.01, 0.012, 0.025);
    for (int l = 0; l < 2; l++) {
        float3 p = rd * (l == 0 ? 90.0 : 220.0);
        float3 id = floor(p);
        float hv = hash13(id + float(l) * 19.7);
        if (hv > 0.97) {
            float3 jit = float3(hash13(id + 1.3), hash13(id + 2.7), hash13(id + 5.1)) - 0.5;
            float d = length(fract(p) - 0.5 - jit * 0.6);
            float3 tint = mix(float3(0.7, 0.8, 1.0), float3(1.0, 0.85, 0.7), hash13(id + 9.9));
            col += tint * smoothstep(0.2, 0.0, d) * (hv - 0.97) / 0.03 * 1.2;
        }
    }
    return col;
}

static float3 shadeNebula(float2 frag, float2 res, constant Uniforms &u) {
    float time = u.resTime.z;
    float3 ro = u.camPos.xyz * 1.7;
    float3 rd = cameraRay(frag, res, ro, float3(0.0), 1.5, 0.0);
    float b = dot(ro, rd);
    float h = b * b - (dot(ro, ro) - 16.0);
    float3 acc = float3(0.0);
    float T = 1.0;
    if (h > 0.0) {
        float sq = sqrt(h);
        float t0 = max(-b - sq, 0.0), t1 = -b + sq;
        int steps = u.scene.z == 0 ? 40 : (u.scene.z == 1 ? 64 : 96);
        float dt = (t1 - t0) / float(steps);
        float t = t0 + dt * hash12(frag);
        for (int i = 0; i < steps; i++) {
            float3 p = ro + rd * t;
            float den = nebulaDensity(p, time);
            if (den > 0.002) {
                float r = length(p);
                float tone = noise3(p * 0.5 + 4.0);
                float3 emis = mix(float3(0.95, 0.18, 0.45), float3(0.10, 0.55, 0.95), smoothstep(0.3, 0.7, tone));
                emis = mix(float3(1.0, 0.7, 0.35), emis, smoothstep(0.6, 2.0, r));
                float light = 2.2 / (1.0 + r * r * 0.9);
                acc += T * emis * den * light * dt * 3.0;
                T *= exp(-den * dt * 4.5);
                if (T < 0.02) break;
            }
            t += dt;
        }
    }
    float tp = -b;
    if (tp > 0.0) {
        float dc = length(ro + rd * tp);
        acc += T * float3(1.0, 0.85, 0.7) * 0.012 / (dc * dc + 0.0015);
    }
    return acc + starField(rd) * T;
}

// ============================================================== frame assembly

static float3 shadeScene(int id, float2 frag, float2 res, constant Uniforms &u) {
    switch (id) {
        case 0: return shadeBulb(frag, res, u);
        case 1: return shadeCanyon(frag, res, u);
        case 2: return shadeSponge(frag, res, u);
        default: return shadeNebula(frag, res, u);
    }
}

static float3 finish(float3 col, float2 frag, float2 res) {
    col = (col * (2.51 * col + 0.03)) / (col * (2.43 * col + 0.59) + 0.14); // ACES fit
    col = pow(clamp(col, 0.0, 1.0), float3(1.0 / 2.2));
    float2 q = frag / res;
    col *= 0.55 + 0.45 * pow(16.0 * q.x * q.y * (1.0 - q.x) * (1.0 - q.y), 0.18);
    return col;
}

static float3 renderPixel(uint2 gid, constant Uniforms &u, int y) {
    float2 res = u.resTime.xy;
    float2 px = float2(gid.x, y);
    int id = kScene;
    float2 org = float2(0.0), lres = res;
    bool border = false;
    if (id == 4) {  // gallery: one scene per quadrant
        float2 halfRes = floor(res * 0.5);
        int qx = px.x >= halfRes.x ? 1 : 0;
        int qy = px.y >= halfRes.y ? 1 : 0;
        id = qx + 2 * qy;
        org = float2(qx, qy) * halfRes;
        lres = float2(qx == 1 ? res.x - halfRes.x : halfRes.x, qy == 1 ? res.y - halfRes.y : halfRes.y);
        float2 l = px - org;
        border = (qx == 1 && l.x < 2.0) || (qy == 1 && l.y < 2.0);
    }
    int spp = max(u.rows.z, 1);
    float3 acc = float3(0.0);
    for (int sy = 0; sy < spp; sy++)
        for (int sx = 0; sx < spp; sx++) {
            float2 off = (float2(sx, sy) + 0.5) / float(spp);
            acc += shadeScene(id, px - org + off, lres, u);
        }
    float3 col = finish(acc / float(spp * spp), px - org + 0.5, lres);
    if (border) col = float3(0.0);
    if (u.flags.x != 0) {
        float3 tint = u.flags.y == 0 ? float3(0.25, 0.6, 1.0) : float3(1.0, 0.55, 0.15);
        col = mix(col, tint, 0.16);
        if (y == u.rows.x && y != 0) col = float3(1.0);
    }
    return col;
}

// Writes one band of rows into the target texture (the frame on GPU 0, a private texture on GPU 1).
kernel void renderTex(texture2d<float, access::write> out [[texture(0)]],
                      constant Uniforms &u [[buffer(0)]],
                      uint2 gid [[thread_position_in_grid]]) {
    if (int(gid.x) >= int(u.resTime.x) || int(gid.y) >= u.rows.y) return;
    int y = int(gid.y) + u.rows.x;
    out.write(float4(renderPixel(gid, u, y), 1.0), uint2(gid.x, y));
}

// ============================================================== galaxy collision (N-body)

// One integration step for bodies [range.x, range.x + range.y) out of range.z: all-pairs
// gravity, tiled through threadgroup memory. Each body gets 4 threads that each sum a
// quarter of every tile, then combine: 4x the threads in flight hides memory latency
// (with ~24k bodies per GPU, one thread per body leaves the D500 starved).
// Results go to this GPU's private copy and to shared system memory (positions at
// [0, n), velocities at [n, 2n)) for the other GPU.
kernel void nbodyStep(device const float4 *posIn [[buffer(0)]],
                      device const float4 *velIn [[buffer(1)]],
                      device float4 *posOut [[buffer(2)]],
                      device float4 *velOut [[buffer(3)]],
                      device float4 *shared [[buffer(4)]],
                      constant uint4 &range [[buffer(5)]],
                      constant float4 &sim [[buffer(6)]],     // x = dt, y = softening^2
                      uint tg [[threadgroup_position_in_grid]],
                      uint lid [[thread_index_in_threadgroup]]) {
    threadgroup float4 tile[256];
    threadgroup float4 partial[256];
    uint n = range.z;
    uint body = lid & 63, slice = lid >> 6;
    uint i = range.x + tg * 64 + body;
    float4 pi = posIn[i];
    float3 acc = float3(0.0);
    for (uint base = 0; base < n; base += 256) {
        tile[lid] = posIn[base + lid];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = slice * 64; k < slice * 64 + 64; k++) {
            float4 pj = tile[k];
            float3 d = pj.xyz - pi.xyz;
            float inv = rsqrt(dot(d, d) + sim.y);
            acc += d * (pj.w * inv * inv * inv);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    partial[lid] = float4(acc, 0.0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (slice != 0) return;
    acc = partial[body].xyz + partial[body + 64].xyz + partial[body + 128].xyz + partial[body + 192].xyz;
    float4 v = velIn[i];
    v.xyz += acc * sim.x;
    float4 p = float4(pi.xyz + v.xyz * sim.x, pi.w);
    posOut[i] = p;
    velOut[i] = v;
    shared[i] = p;
    shared[n + i] = v;
}

struct PointOut {
    float4 position [[position]];
    float size [[point_size]];
    float3 color;
};

vertex PointOut pointVertex(uint vid [[vertex_id]],
                            device const float4 *pos [[buffer(0)]],
                            device const float4 *vel [[buffer(1)]],
                            constant float4x4 &viewProj [[buffer(2)]],
                            constant float4 &style [[buffer(3)]],   // x = size, y = intensity, z = n/2, w = pixel scale
                            constant float4 &extra [[buffer(4)]]) { // x = tint by GPU, y = bodies on GPU 0
    PointOut o;
    float4 p = pos[vid];
    o.position = viewProj * float4(p.xyz, 1.0);
    float k = smoothstep(0.3, 1.8, length(vel[vid].xyz));
    bool second = float(vid) >= style.z;
    float3 slow = second ? float3(1.0, 0.42, 0.62) : float3(1.0, 0.62, 0.30);
    float3 fast = second ? float3(0.62, 0.78, 1.0) : float3(0.55, 0.85, 1.0);
    float3 c = mix(slow, fast, k);
    if (extra.x > 0.5) c = float(vid) < extra.y ? float3(0.25, 0.6, 1.0) : float3(1.0, 0.55, 0.15);
    o.color = p.w > 0.1 ? float3(0.0) : c * style.y;
    o.size = clamp(style.x * style.w * 6.0 / max(o.position.w, 0.1), 1.0, style.x * style.w * 4.0);
    return o;
}

fragment float4 pointFragment(PointOut in [[stage_in]], float2 pc [[point_coord]]) {
    float2 d = pc * 2.0 - 1.0;
    float r2 = dot(d, d);
    if (r2 > 1.0) discard_fragment();
    return float4(in.color * exp(-r2 * 3.0), 1.0);
}

kernel void galaxyTonemap(texture2d<float, access::read> hdr [[texture(0)]],
                          texture2d<float, access::write> out [[texture(1)]],
                          constant float4 &params [[buffer(0)]],   // x = exposure
                          uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) return;
    float3 c = 1.0 - exp(-hdr.read(gid).rgb * params.x);
    float2 q = float2(gid) / float2(out.get_width(), out.get_height());
    c += mix(float3(0.012, 0.012, 0.03), float3(0.0, 0.0, 0.005), q.y);
    out.write(float4(pow(clamp(c, 0.0, 1.0), float3(1.0 / 2.2)), 1.0), gid);
}
