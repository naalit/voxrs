#version 330 core

in vec3 pos;
in vec3 nor;
out vec3 normal;
out vec3 frag_pos;

uniform mat4 proj_mat;
uniform mat4 model;

void main() {
    gl_Position = proj_mat * model * vec4(pos, 1.0);
    frag_pos = pos;
    normal = nor;
}
