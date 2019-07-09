#version 330 core

out vec4 frag_color;

in vec3 frag_pos;
in vec3 normal;
flat in uint mat_index;

uint encode_normal(vec3 n) {
    uint ret = 0u;
    ret |= uint(abs(n.x));
    ret |= uint(abs(n.y)) << 1u;

    ret |= uint(min(n.x,0.0)) << 2u;
    ret |= uint(min(n.y,0.0)) << 2u;
    ret |= uint(min(n.z,0.0)) << 2u;

    return ret;
}

vec3 decode_normal(uint n) {
    return sign(0.5-float((n >> 2u) & 1u))* vec3(
        float(n & 1u),
        float((n >> 1u) & 1u),
        float(1u - (n & 1u)) * float(1u - ((n >> 1u) & 1u))
        );
}

void main() {
    uint w = (mat_index << 3u) | encode_normal(normal);
    frag_color = vec4(frag_pos, uintBitsToFloat(w));
}
