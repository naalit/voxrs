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
    gl_Position = proj_mat * model * vec4(pos, 1.0);
    frag_pos = (model * vec4(pos, 1.0)).xyz;
    normal = nor;
    mat_index = mat;
}
