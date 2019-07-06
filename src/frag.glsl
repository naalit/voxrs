#version 450

out vec4 frag_color;

in vec3 frag_pos;
in vec3 normal;

// #define WIREFRAME
// #define WF_LIGHTING

void main() {
    vec3 col = vec3(0);
    float NoL = dot(normal, normalize(vec3(0.2, 0.9, 0.3)));
    col += max(0.0, NoL );
    col += max(0.01, -0.1 * NoL ); // Fake bounce, inspired by IQ's three light model
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
