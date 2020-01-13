#version 330 core

in vec3 pos;
in vec3 nor;
in uint mat;
out vec3 normal;
out vec3 frag_pos;
flat out uint mat_index;

uniform mat4 proj_mat;
uniform mat4 model;

void main() {
    vec4 p = model * vec4(pos, 1.0);
    gl_Position = proj_mat * p;
    frag_pos = p.xyz / p.w;
    normal = nor;
    mat_index = mat;
}
