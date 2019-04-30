#version 330 core

in vec2 pos;
// in vec3 col;

// out vec3 vertColor;

void main() {
    gl_Position = vec4(pos, 0.0, 1.0);
    // vertColor = col;
}
