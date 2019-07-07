#version 450

out vec4 frag_color;

in vec3 frag_pos;
in vec3 normal;
flat in uint mat_index;

uniform vec3 camera_pos;

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

void main() {
    vec3 col = vec3(0);
    vec3 sun_dir = normalize(vec3(0.2, 0.9, 0.3));
    float NoL = dot(normal, sun_dir);
    // Roughly IQ's three-light model for fake GI
    vec3 sun_color = vec3(1.3, 1.2, 1.2);
    vec3 sky_color = vec3(0.9, 0.9, 1.1);
    col += max(0.0, NoL ) * sun_color;
    col += max(0.0, 0.5 * dot(abs(normal), normalize(vec3(0.5, 0.2, 0.5))) ) * sky_color;
    col += max(0.01, -0.1 * NoL ) * pow(sun_color, vec3(2.2));
    MatData mat = materials[mat_index];
    col *= pow(mat.color, vec3(2.2)); // sRGB
    col = applyFog(col, length(frag_pos - camera_pos), camera_pos, normalize(frag_pos - camera_pos), sun_dir);
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
    frag_color = vec4(col, 1.0);
}
