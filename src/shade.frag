#version 450

// The shading pass after the G-Buffer is filled

const uint NUM_CASCADES = 4;
const float SHADOW_FAC = 4.0;

out vec4 frag_color;
in vec2 uv;

uniform vec3 camera_pos;
uniform mat4 proj_mat;
uniform light_buf {
  mat4 light_mats[NUM_CASCADES];
};
uniform sampler2D gbuff;
uniform sampler2DArrayShadow shadow_map;
uniform vec3 sun_dir;
uniform uvec2 resolution;

struct MatData {
    vec3 color;
    float roughness;
    float trans;
    float metal;
    float ior;
    float nothing; // Just buffer to pack it in right
};

layout(std140) buffer mat_buf {
    MatData materials[];
};

// Basic shadow mapping
float shadow(vec3 normal, vec3 pos) {
    float far = 1024;
    float z = length(pos - camera_pos);
    uint i = uint(float(NUM_CASCADES) * pow(z/far, 1.0/SHADOW_FAC));

    // Slope-scaled depth bias
    float bias = 0.005 * tan(acos(dot(normal, sun_dir))); //0.0001

    vec4 map_coord = light_mats[i] * vec4(pos, 1.0);

    float depth = (map_coord.z-bias) / map_coord.w * 0.5 + 0.5;
    vec2 coords = map_coord.xy / map_coord.w  * 0.5 + 0.5;

    return texture(shadow_map, vec4(coords, float(i), depth));
}

// From IQ: https://iquilezles.org/www/articles/fog/fog.htm
vec3 applyFog( in vec3 rgb, // original color of the pixel
               in float dist, // camera to point distance
               in vec3 rayOri, // camera position
               in vec3 rayDir, // camera to point vector
               in vec3 sunDir ) { // sun light direction
    float c = 0.008; // Overall fog density
    float b = 0.1; // Altitude falloff
    // float c = a/b;

    float fogAmount = c * exp(-rayOri.y*b) * (1.0-exp( -dist*rayDir.y*b ))/rayDir.y;
    float sunAmount = max( dot( rayDir, sunDir ), 0.0 );
    vec3 fogColor = mix( vec3(0.5,0.6,0.7), // bluish
        vec3(1.0,0.9,0.7), // yellowish
        pow(sunAmount,8.0) );
    return mix( rgb, fogColor, fogAmount );
}

vec3 decode_normal(uint n) {
    return sign(0.5-float((n >> 2u) & 1u))* vec3(
        float(n & 1u),
        float((n >> 1u) & 1u),
        float(1u - (n & 1u)) * float(1u - ((n >> 1u) & 1u))
        );
}

#include <sky.glsl>

#define PI 3.1415926535
const float R2PI = sqrt(2.0/PI);
const float IPI = 1.0 / PI;
#define saturate(x) clamp(x, 0.0, 1000.0)
float sqr(float x) { return x*x; }

float D(float NoH, float roughness) {
    float a = sqr(roughness);
    float a2 = sqr(a);
    float denom = PI * sqr(sqr(NoH) * (a2 - 1.0) + 1.0);
    return a2 / denom;
}
float G1(float NoV, float k) {
    float denom = NoV * (1.0 - k) + k;
    return NoV / denom;
}
float G(float NoL, float NoV, float roughness) {
    float k = sqr(roughness + 1) / 8.0;
    return G1(NoL, k) * G1(NoV, k);
}
float F(float VoH, float ior) {
    float f0 = sqr((1.0-ior)/(1.0+ior));
    return f0 + (1.0 - f0) * exp2((-5.55473*VoH - 6.98316) * VoH);
}
// Just specular
vec3 bsdf(in vec3 V, in vec3 L, in vec3 N, in MatData mat) {
    vec3 H = normalize(V+L);
    float NoL = dot(N, L);
    float NoV = dot(N, V);

    float nom = D(dot(N, H), mat.roughness)
        * F(dot(V, H), mat.ior)
        * G(NoL, NoV, mat.roughness);

    // Prevent division by zero, even when the GPU uses 8-bit arithmetic
    float denom = 4.0 * NoV;// * NoL;

    return max(vec3(0.01) * max(nom,0.0) / denom,0.0);
}

vec3 shade(vec3 rd, vec3 normal, MatData mat, vec3 pos) {
    vec3 sun_color = pow(vec3(0.7031,0.4687,0.1055), vec3(1.0 / 4.2));
    vec3 sky_color = pow(vec3(0.3984,0.5117,0.7305), vec3(1.0 / 4.2));

    float sha = shadow(normal, pos);

    vec3 col = sha * sun_color * smoothstep(0.0, 0.1, sun_dir.y) * saturate(dot(normal, sun_dir));//saturate(bsdf(-rd, sun_dir, normal, mat));
    col += sky_color * 0.2 * saturate(0.5 + 0.5*normal.y + 0.2*normal.x);//mat.color * IPI * length(abs(normal) * vec3(0.7, 1.0, 0.85));//bsdf(-rd, normalize(normal * vec3(1, 0, 1)), normal, mat);
    col += sha * pow(sun_color, vec3(1.2)) * 0.2 * smoothstep(0.0, 0.1, sun_dir.y) * saturate(dot(normal, normalize(sun_dir * vec3(-1,0,-1))));//saturate(bsdf(-rd, -sun_dir, normal, mat));

    col *= IPI * mat.color;
    if (mat.roughness < 0.2) {
        vec3 r = reflect(rd,normal);
        col += 0.25 * sky(pos, r) * min(vec3(1.0),bsdf(-rd, normalize(r), normal, mat));
    }
    col = applyFog(col, length(pos-camera_pos), camera_pos, rd, sun_dir);
    return col;
}

void main() {
    vec4 g = texelFetch(gbuff, ivec2(gl_FragCoord.xy), 0);
    vec3 frag_pos = g.xyz;
    uint w = floatBitsToUint(g.w);
    uint mat_index = w >> 3u;
    vec3 normal = decode_normal(w);

    vec3 col = vec3(0);
    float a = 1.0;
    vec2 uva = uv;
    uva.x *= float(resolution.x) / float(resolution.y);
    if (length(uva) < 0.005) {
        col = vec3(1);
    } else if (mat_index == 0u) {
        vec4 cd = inverse(proj_mat) * vec4(uv,1,1);
        col = sky(camera_pos, normalize(cd.xyz/cd.w - camera_pos));
    } else {
        vec3 rd = normalize(frag_pos - camera_pos);
        MatData mat = materials[mat_index];
        col = shade(rd, normal, mat, frag_pos);
        a = 1.0 - mat.trans;
    }

    //col = vec3(1) * texture(shadow_map, vec4(uv, 0.0, 0.0));

    // Vignette
    // col *= smoothstep(-0.5,0.8,1.0-length(pow(abs(uv),vec2(2.0))));
    // No gamma correction, we're using a sRGB framebuffer
    frag_color = vec4(col, a);
}
