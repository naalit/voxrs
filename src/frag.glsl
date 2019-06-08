#version 450

uniform uvec2 resolution;

uniform vec3 camera_dir;
uniform vec3 camera_up;
uniform vec3 camera_right;
uniform vec3 camera_pos;

out vec4 frag_color;

#include <octree.glsl>
#include <shade.glsl>

void main() {
    // -1 to 1, aspect corrected
    vec2 uv = 2.0 * gl_FragCoord.xy / vec2(resolution.y) - 1.0;

    vec3 rd = normalize(
        camera_dir  // The point on the film we're looking at
        + camera_up * uv.y // Offset this much up
        + camera_right * uv.x // And this much right
    );
    vec3 ro = camera_pos;//vec3(0.0,0.0,0.0);

    vec2 t;
    int i;
    vec3 pos;
    bool b = trace(ro,rd,t,i,pos);

    frag_color = vec4(vec3(b) * shade(ro,rd,t,pos),1.0);// * vec4(t.y*0.001);//vec4(i)/128.0;//*/vec4(i)/32.0;///*vec4(b)  vec4(i)/64.0;/*/vec4(t*0.01,0.0,1.0);
}
