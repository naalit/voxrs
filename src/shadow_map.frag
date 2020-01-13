#version 330 core

out float frag_depth;

// These things are here so we can use the GBuffer vertex shader
in vec3 frag_pos;
in vec3 normal;
flat in uint mat_index;

void main() {
  frag_depth = gl_FragDepth = gl_FragCoord.z;
}
