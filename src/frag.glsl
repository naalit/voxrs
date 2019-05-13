#version 450

uniform float iTime;
uniform vec2 iResolution;
uniform vec3 cameraPos;
uniform vec3 cameraDir;
uniform vec3 cameraUp;

#define CHUNK_SIZE 16

struct MatData {
    vec3 color;
    float roughness;
    float trans;
    float metal;
    float ior;
    float nothing; // Just buffer to pack it in right
};

layout(std140) buffer mat_list {
    MatData materials[];
};

uniform usampler3D chunks;
uniform usampler3D blocks;
uniform vec3 chunk_origin;

out vec4 fragColor;

void mainImage( out vec4 fragColor, in vec2 fragCoord );

// #define COMPARE

// #define REINHARD
// #define HEJL_DAWSON
// #define UNCHARTED

#ifdef UNCHARTED
float A = 0.15;
float B = 0.50;
float C = 0.10;
float D = 0.20;
float E = 0.02;
float F = 0.30;
float W = 11.2;

vec3 uc2_tonemap(vec3 x) {
    return ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F))-E/F;
}
#endif

vec3 tonemap(in vec3 col) {
    vec3 ret = pow(col,vec3(2.2));
    #ifdef COMPARE
    if (gl_FragCoord.x > iResolution.x / 2.0)
    #endif
    {
    #ifdef REINHARD
    ret /= 1.0 + length(ret);
    #else
    #ifdef HEJL_DAWSON
    vec3 x = max(vec3(0.0),ret-0.004);
    ret = (x*(6.2*x+0.5))/(x*(6.2*x+1.7)+0.06);
    #else
    #ifdef UNCHARTED
    float bias = 2.0;
    vec3 cur = uc2_tonemap(bias*ret);
    vec3 white_scale = 1.0 / uc2_tonemap(vec3(W));
    ret = cur * white_scale;
    #endif
    #endif
    #endif
    }
    return ret;
}

void main() {
    mainImage(fragColor, gl_FragCoord.xy);
    fragColor = vec4(tonemap(fragColor.xyz),1.0);//pow(fragColor, vec4(2.2)); // mon2lin
}


// -----------------------------------------------------------------------------------------


// The maximum iterations for voxel traversal
const int MAX_ITER = 256;

#define PI 3.1415926535
const float IPI = 1./PI;
const float R2PI = sqrt(2./PI);

float sqr(float x) { return x*x; }
#define saturate(x) clamp(x,0.,1.)

vec2 isect(in vec3 pos, in float size, in vec3 ro, in vec3 rd, out vec3 tmid, out vec3 tmax) {
    vec3 mn = pos - 0.5 * size;
    vec3 mx = mn + size;
    vec3 t1 = (mn-ro) / rd;
    vec3 t2 = (mx-ro) / rd;
    vec3 tmin = min(t1, t2);
    tmax = max(t1, t2);
    tmid = (pos-ro)/rd; // tmax;
    return vec2(max(tmin.x, max(tmin.y, tmin.z)), min(tmax.x, min(tmax.y, tmax.z)));
}


// -----------------------------------------------------------------------------------------


const int LOG2_CHUNK = int(log2(CHUNK_SIZE));
int scene_size;

uint getVoxel(ivec3 pos) {
    if (any(lessThan(pos,ivec3(0))) || any(greaterThanEqual(pos,ivec3(scene_size))))
        return 121212; // Bigger than the max for u16, so it will never be this for real
    ivec3 chunk = /*pos >> LOG2_CHUNK; /*/ pos / CHUNK_SIZE;
    ivec3 in_chunk = /*pos & (CHUNK_SIZE - 1); /*/ pos % CHUNK_SIZE;
    uvec3 offset = texelFetch(chunks, chunk, 0).zyx;
    return texelFetch(blocks, ivec3(offset)*CHUNK_SIZE + in_chunk, 0).r;
}

// Regular grid
uint trace(in vec3 ro, in vec3 rd, in uint ignore, out vec2 t, out vec3 pos, inout int iter, out float size, out vec3 normal) {
    size = 1.0;

    ivec3 total_size = textureSize(blocks, 0);
    scene_size = total_size.x;
    float root_size = float(total_size.x) * 0.5;

    vec3 tmid,
        tmax;
    // We translate it to "grid space" first, so pos = vec3(0) is actually chunk_origin in world coordinates
    vec3 ro_chunk = ro - chunk_origin + root_size;
    pos = ro_chunk;
    ivec3 ipos = ivec3(floor(pos));
    ivec3 istep = ivec3(sign(rd));
    vec3 tdelta = abs(1.0 / rd);
    vec3 sideDist = (sign(rd) * (vec3(ipos) - ro_chunk) + (sign(rd) * 0.5) + 0.5) * tdelta;
    bvec3 mask;

    iter = MAX_ITER / iter;
    while (iter --> 0) {
        if (any(lessThan(ipos,ivec3(0))) || any(greaterThanEqual(ipos,ivec3(scene_size))))
            return 0;
        ivec3 chunk = ipos / CHUNK_SIZE;
        uvec3 offset = texelFetch(chunks, chunk, 0).zyx;
        // Skip chunk w/o checking voxels in between
        if (offset.x == 255) {
            do {
                mask = lessThanEqual(sideDist.xyz, min(sideDist.yzx, sideDist.zxy));
                sideDist += vec3(mask) * tdelta;
                ipos += ivec3(mask) * istep;
            } while (ipos / CHUNK_SIZE == chunk);
            continue;
        }
        uint voxel =
            texelFetch(blocks, ivec3(offset)*CHUNK_SIZE+(ipos % CHUNK_SIZE),0).r;
            // getVoxel(ipos);
        if (voxel != 0 && voxel != ignore) {
            t = isect(vec3(ipos) + 0.5, size, ro_chunk, rd, tmid, tmax);
            normal = vec3(mask);
            pos = vec3(ipos) + 0.5 + chunk_origin - root_size; // Translate it back to world space
            return voxel;
        }

        mask = lessThanEqual(sideDist.xyz, min(sideDist.yzx, sideDist.zxy));
        sideDist += vec3(mask) * tdelta;
        ipos += ivec3(mask) * istep;
    }
    return 0;
}

float shadow(in vec3 ro, in vec3 rd, in uint ignore) {
    ivec3 total_size = textureSize(blocks, 0);
    scene_size = total_size.x;
    float root_size = float(total_size.x) * 0.5;

    // We translate it to "grid space" first, so pos = vec3(0) is actually chunk_origin in world coordinates
    vec3 ro_chunk = ro - chunk_origin + root_size;

    ivec3 ipos = ivec3(floor(ro_chunk));
    ivec3 istep = ivec3(sign(rd));
    vec3 tdelta = abs(1.0 / rd);
    vec3 sideDist = (sign(rd) * (vec3(ipos) - ro_chunk) + (sign(rd) * 0.5) + 0.5) * tdelta;
    bvec3 mask;

    int iter = MAX_ITER / 4;
    while (iter --> 0) {
        if (any(lessThan(ipos,ivec3(0))) || any(greaterThanEqual(ipos,ivec3(scene_size))))
            return 1.0;
        ivec3 chunk = ipos / CHUNK_SIZE;
        uvec3 offset = texelFetch(chunks, chunk, 0).zyx;
        // Skip chunk w/o checking voxels in between
        if (offset.x == 255) {
            do {
                mask = lessThanEqual(sideDist.xyz, min(sideDist.yzx, sideDist.zxy));
                sideDist += vec3(mask) * tdelta;
                ipos += ivec3(mask) * istep;
            } while (ipos / CHUNK_SIZE == chunk);
            continue;
        }
        uint voxel =
            texelFetch(blocks, ivec3(offset)*CHUNK_SIZE+(ipos % CHUNK_SIZE),0).r;
        // if (voxel == 121212)
        //     return 0.0;
        if (voxel != 0 && voxel != ignore) {
            return 0.0;
        }

        mask = lessThanEqual(sideDist.xyz, min(sideDist.yzx, sideDist.zxy));
        sideDist += vec3(mask) * tdelta;
        ipos += ivec3(mask) * istep;
    }
    return 1.0;
}

uint trace(in vec3 ro, in vec3 rd, out vec2 t, out vec3 pos, inout int iter, out float size, out vec3 normal) {
    return trace(ro,rd,0,t,pos,iter,size,normal);
}


// -----------------------------------------------------------------------------------------


#include <sky.glsl>
#include <bsdf.glsl>


// -----------------------------------------------------------------------------------------


// By Dave_Hoskins https://www.shadertoy.com/view/4djSRW
vec3 hash33(vec3 p3)
{
	p3 = fract(p3 * vec3(.1031, .1030, .0973));
    p3 += dot(p3, p3.yxz+19.19);
    return fract((p3.xxy + p3.yxx)*p3.zyx);

}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 uv = fragCoord / iResolution.xy;
    uv *= 2.;
    uv -= 1.;
    uv.y *= iResolution.y / iResolution.x;
    if (length(uv) < 0.006 && length(uv) > 0.005) {
        fragColor = vec4(1.0);
        return;
    }

    vec3 ro = cameraPos;
    vec3 up = cameraUp;
    vec3 right = normalize(cross(cameraDir, cameraUp));
    vec3 rd = cameraDir;
    float FOV = 1.0; // Not actual FOV, just a multiplier
    rd += FOV * up * uv.y;
    rd += FOV * right * uv.x;
    rd = normalize(rd);

    vec2 t;
    vec3 pos;
    float size;
    int iter = 1;
    vec3 n;

    uint mat = trace(ro, rd, t, pos, iter, size, n);
    vec3 col = mat != 0 ? shade(mat, ro, rd, t, iter, pos, n) : sky(ro, rd);
    // col = vec3(iter) / vec3(MAX_ITER);

    fragColor = vec4(col,1.0);
}
