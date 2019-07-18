#version 450

// The shading pass after the G-Buffer is filled

out vec4 frag_color;
in vec2 uv;

uniform vec3 camera_pos;
uniform mat4 proj_mat;
uniform sampler2D gbuff;
uniform vec3 sun_dir;

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

// #define WIREFRAME
// #define WF_LIGHTING

// From IQ: https://iquilezles.org/www/articles/fog/fog.htm
vec3 applyFog( in vec3 rgb, // original color of the pixel
               in float dist, // camera to point distance
               in vec3 rayOri, // camera position
               in vec3 rayDir, // camera to point vector
               in vec3 sunDir ) { // sun light direction
    float c = 0.05;
    float b = 0.05;

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

#if 0
vec3 bsdf(in vec3 v, in vec3 n, in vec3 l, in MatData mat) {
    // See "Real Shading in Unreal Engine 4"

    vec3 diff = mat.color * IPI;

    vec3 h = normalize(v+l);

    float nDotL = dot(n,l);
    float nDotV = dot(n,v);
    float nDotH = dot(n,h);
    float vDotH = dot(v,h);

    float f0 = sqr((1-mat.ior)/(1+mat.ior));

    float a2 = mat.roughness * mat.roughness;
    a2 *= a2; // Is this right?
    float D = a2 / (PI * sqr(sqr(nDotH) * (a2 - 1.0) + 1.0));
    float k = sqr(mat.roughness+1.0) / 8.0;
    float G = nDotV*nDotL / ( (nDotV*(1.0-k)+k) * (nDotL*(1.0-k)+k));
    float F = f0 + (1.0 - f0) * exp2(vDotH * (-5.55473*vDotH-6.98316));
    float spec = D * F * G / (4.0 * nDotL * nDotV);

    return saturate(spec+diff);
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

    vec3 cspec = vec3(0.5);
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

void main() {
    vec4 g = texelFetch(gbuff, ivec2(gl_FragCoord.xy), 0);
    vec3 frag_pos = g.xyz;
    uint w = floatBitsToUint(g.w);
    uint mat_index = w >> 3u;
    vec3 normal = decode_normal(w);

    vec3 col = vec3(0);
    float a = 1.0;
    if (mat_index == 0u) {
        vec4 cd = inverse(proj_mat) * vec4(uv,1,1);
        col = sky(camera_pos, normalize(cd.xyz/cd.w - camera_pos));
    } else {
        vec3 rd = normalize(frag_pos - camera_pos);
        // Roughly IQ's three-light model for fake GI
        vec3 sun_color = vec3(1.3, 1.2, 1.2);
        vec3 sky_color = vec3(0.9, 0.9, 1.1);
        MatData mat = materials[mat_index];
        col += sun_color * bsdf(-rd, sun_dir, normal, mat);
        col += sky_color * bsdf(-rd, normalize(vec3(0.5, 0.2, 0.5)), abs(normal), mat);
        col += pow(sun_color, vec3(1.2)) * saturate(bsdf(-rd, -sun_dir, normal, mat));
        col = applyFog(col, length(frag_pos - camera_pos), camera_pos, rd, sun_dir);
        a = 1.0 - mat.trans;
        // col = pow(col, vec3(2.2)); // sRGB
        #ifdef WIREFRAME
        vec3 uvw = fract(frag_pos);
        float t = max(uvw.x, max(uvw.y, uvw.z));
        uvw += abs(normal)*0.5;
        float q = min(uvw.x, min(uvw.y, uvw.z));
        if (t < 0.99 && q > 0.01)
            discard;
        #ifndef WF_LIGHTING
        col = vec3(1.0);
        #endif
        #endif
    }
    frag_color = vec4(pow(col, vec3(2.2)), a);
}
