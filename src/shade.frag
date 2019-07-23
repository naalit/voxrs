#version 450

// The shading pass after the G-Buffer is filled

out vec4 frag_color;
in vec2 uv;

uniform vec3 camera_pos;
uniform mat4 proj_mat;
uniform sampler2D gbuff;
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

// From IQ: https://iquilezles.org/www/articles/fog/fog.htm
vec3 applyFog( in vec3 rgb, // original color of the pixel
               in float dist, // camera to point distance
               in vec3 rayOri, // camera position
               in vec3 rayDir, // camera to point vector
               in vec3 sunDir ) { // sun light direction
    float c = 0.08; // Overall fog density
    float b = 0.05; // Altitude falloff

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

#if 1
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
vec3 bsdf(in vec3 V, in vec3 L, in vec3 N, in MatData mat) {
    vec3 H = normalize(V+L);
    float NoL = dot(N, L);
    float NoV = dot(N, V);

    float nom = D(dot(N, H), mat.roughness)
        * F(dot(V, H), mat.ior)
        * G(NoL, NoV, mat.roughness);

    // Prevent division by zero, even when the GPU uses 8-bit arithmetic
    float denom = 4.0 * NoV;// * NoL;

    return max(vec3(0.01) * max(nom,0.0) / denom,0.0)
        + mat.color * max(0.0,IPI * max(NoL,0.0) * (1.0 - F(dot(V, H), mat.ior)));
}
#else
vec3 bsdf(
    vec3 v,		// direction from vertex to view
	vec3 l,		// direction from vertex to light
	vec3 n,		// macro surface normal
    MatData mat
) {
	// half vector
	vec3 h = normalize( l + v );

	// dot
	float dot_n_h = max( abs( dot( n, h ) ), 0.001 );
	float dot_n_v = max( abs( dot( n, v ) ), 0.001 );
	float dot_n_l = max( abs( dot( n, l ) ), 0.001 );
	float dot_h_v = max( abs( dot( h, v ) ), 0.001 ); // dot_h_v == dot_h_l

	// Geometric Term
#if 0
    // Cook-Torrance
    //          2 * ( N dot H )( N dot L )    2 * ( N dot H )( N dot V )
	// min( 1, ----------------------------, ---------------------------- )
	//                 ( H dot V )                   ( H dot V )
	float g = 2.0 * dot_n_h / dot_h_v;
	float G = min( min( dot_n_v, dot_n_l ) * g, 1.0 );
#else
    // Implicit
    float G = dot_n_l * dot_n_v;
#endif

    // Normal Distribution Function ( cancel 1 / pi )
#if 1
 	// Beckmann distribution
	//         ( N dot H )^2 - 1
	//  exp( ----------------------- )
	//         ( N dot H )^2 * m^2
	// --------------------------------
	//         ( N dot H )^4 * m^2
    float sq_nh   = dot_n_h * dot_n_h;
	float sq_nh_m = sq_nh * ( mat.roughness * mat.roughness );
	float D = exp( ( sq_nh - 1.0 ) / sq_nh_m ) / ( sq_nh * sq_nh_m );
#else
    // Blinn distribution
    float shininess = 2.0 / ( m * m ) - 2.0;
    float D = ( shininess + 2.0 ) / 2.0 * pow( dot_n_h, shininess );
#endif

    vec3 cspec = vec3(0.0);
    vec3 cdiff = mat.color * IPI;
    float clight = 1.0;

	// Specular Fresnel Term : Schlick approximation
	// F0 + ( 1 - F0 ) * ( 1 - ( H dot V ) )^5
	vec3 Fspec = cspec + ( 1.0  - cspec ) * pow( 1.0 - dot_h_v, 5.0 );

	// Diffuse Fresnel Term : violates reciprocity...
	// F0 + ( 1 - F0 ) * ( 1 - ( N dot L ) )^5
	vec3 Fdiff = cspec + ( 1.0  - cspec ) * pow( 1.0 - dot_n_l, 5.0 );

	// Cook-Torrance BRDF
	//          D * F * G
	// ---------------------------
	//  4 * ( N dot V )( N dot L )
	vec3 brdf_spec = Fspec * D * G / ( dot_n_v * dot_n_l * 4.0 );

	// Lambertian BRDF ( cancel 1 / pi )
	vec3 brdf_diff = cdiff * ( 1.0 - Fdiff );

	// Punctual Light Source ( cancel pi )
	return ( brdf_spec + brdf_diff ) * clight * dot_n_l;
}
#endif

/*

uniform vec3 camera_pos;
in vec3 frag_pos;
flat in uint mat_index;

vec3 rd = normalize(frag_pos - camera_pos);
Material mat = materials[mat_index];
frag_color = brdf(-rd, sun_dir, normal, mat);

*/

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
        // Roughly IQ's three-light model for fake GI
        vec3 sun_color = pow(vec3(0.7031,0.4687,0.1055), vec3(1.0 / 4.2)) * 1.5;
        vec3 sky_color = pow(vec3(0.3984,0.5117,0.7305), vec3(1.0 / 4.2));
        MatData mat = materials[mat_index];
        col += sun_color * smoothstep(0.0, 0.1, sun_dir.y) * saturate(bsdf(-rd, sun_dir, normal, mat));
        col += sky_color * mat.color * IPI * length(abs(normal) * vec3(0.7, 1.0, 0.85));//bsdf(-rd, normalize(normal * vec3(1, 0, 1)), normal, mat);
        col += pow(sun_color, vec3(1.2)) * smoothstep(0.0, 0.1, sun_dir.y) * saturate(bsdf(-rd, -sun_dir, normal, mat));
        if (mat.roughness < 0.2) {
            vec3 r = reflect(rd,normal);
            col += 0.25 * sky(frag_pos, r) * min(vec3(1.0),bsdf(-rd, normalize(r), normal, mat));
        }
        col = applyFog(col, length(frag_pos - camera_pos), camera_pos, rd, sun_dir);
        a = 1.0 - mat.trans;
    }
    frag_color = vec4(pow(col, vec3(2.2)), a);
}
