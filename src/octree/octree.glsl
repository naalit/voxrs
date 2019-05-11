// We'll need to change the max later
#define MAX_LEVELS 8
uniform int levels;

buffer octree {
    dvec4 nodes[];
};

struct Node {
    bool[8] leaf;
    uint[8] pointer;
};

// The equivalent function in Rust follows this one closely to allow for testing of this one,
//  so only update both at once
#if 0
Node decode(dvec4 source) {
    bool[8] leaf;
    // Reverse order - least significant, most significant
    uvec2 one = unpackDouble2x32(source.x);
    for (int i = 0; i < 8; i++) {
        leaf[i] = one.y >= (1 << (31-i));
        one.y = one.y % (1 << (31-i));
    }
    uint[8] pointer;
    // Reverse order again
    uvec2 two = unpackDouble2x32(source.y);
    uvec2 three = unpackDouble2x32(source.z);

    pointer[0] = bitfieldExtract(two.y, 16, 16); // Most significant
    pointer[1] = bitfieldExtract(two.y, 0, 16); // Least significant
    pointer[2] = bitfieldExtract(two.x, 16, 16); // Most significant
    pointer[3] = bitfieldExtract(two.x, 0, 16); // Least significant

    pointer[4] = bitfieldExtract(three.y, 16, 16); // Most significant
    pointer[5] = bitfieldExtract(three.y, 0, 16); // Least significant
    pointer[6] = bitfieldExtract(three.x, 16, 16); // Most significant
    pointer[7] = bitfieldExtract(three.x, 0, 16); // Least significant

    pointer[0] |= bitfieldExtract(one.y, 25-8, 7) << 16; // bits 0-7 are used up
    pointer[1] |= bitfieldExtract(one.y, 25-15, 7) << 16;
    pointer[2] |= bitfieldExtract(one.y, 25-22, 7) << 16;

    pointer[3] |= bitfieldExtract(one.y, 29-29, 3) << 20; // 16+4=20; 29+3=32
    pointer[3] |= bitfieldExtract(one.x, 28-0, 4) << 16;

    pointer[4] |= bitfieldExtract(one.x, 25-4, 7) << 16;
    pointer[5] |= bitfieldExtract(one.x, 25-11, 7) << 16;
    pointer[6] |= bitfieldExtract(one.x, 25-18, 7) << 16;
    pointer[7] |= bitfieldExtract(one.x, 25-25, 7) << 16; // 25 + 7 = 32

    return Node(leaf, pointer);
}
#else
uint bitfield(uint source, int offset, int bits) {
    return (source & (((1 << bits) - 1) << offset)) >> offset;
}

Node decode(dvec4 source) {
    uvec2 one = unpackDouble2x32(source.x);
    uvec2 two = unpackDouble2x32(source.y);
    uvec2 three = unpackDouble2x32(source.z);

    return Node(
        bool[](
            (one.y & (1 << 31)) != 0,
            (one.y & (1 << 30)) != 0,
            (one.y & (1 << 29)) != 0,
            (one.y & (1 << 28)) != 0,
            (one.y & (1 << 27)) != 0,
            (one.y & (1 << 26)) != 0,
            (one.y & (1 << 25)) != 0,
            (one.y & (1 << 24)) != 0
            ),
        uint[](
            bitfield(two.y, 16, 16) | (bitfield(one.y, 25-8, 7) << 16),
            bitfield(two.y, 0, 16) | (bitfield(one.y, 25-15, 7) << 16),
            bitfield(two.x, 16, 16) | (bitfield(one.y, 25-22, 7) << 16),
            bitfield(two.x, 0, 16) | (bitfield(one.y, 29-29, 3) << 20) | (bitfield(one.x, 28-0, 4) << 16),
            bitfield(three.y, 16, 16) | (bitfield(one.x, 25-4, 7) << 16),
            bitfield(three.y, 0, 16) | (bitfield(one.x, 25-11, 7) << 16),
            bitfield(three.x, 16, 16) | (bitfield(one.x, 25-18, 7) << 16),
            bitfield(three.x, 0, 16) | (bitfield(one.x, 25-25, 7) << 16)
            )
        );
}
#endif

// The idx has bits `x,y,z`, from most to least significant
uint uidx(vec3 idx) {
    uint ret = 0u;
    ret |= uint(idx.x > 0.0) << 2;
    ret |= uint(idx.y > 0.0) << 1;
    ret |= uint(idx.z > 0.0);
    return ret;
}

bool leaf(Node parent, uint idx) {
    return parent.leaf[idx];
}

bool leaf(Node parent, vec3 idx) {
    return parent.leaf[uidx(idx)];
}

Node voxel(Node parent, uint idx) {
    return decode(nodes[parent.pointer[idx]]);
}

Node voxel(Node parent, vec3 idx) {
    return decode(nodes[parent.pointer[uidx(idx)]]);
}


// The size of the scene. Don't change unless you change the distance function
const float root_size = 16.*2.*2.;

struct ST {
    Node parent;
    vec3 pos;
	int scale; // size = root_size * exp2(float(-scale));
    vec3 idx;
    float h;
} stack[MAX_LEVELS];

int stack_ptr = 0; // Next open index
void stack_reset() { stack_ptr = 0; }
void stack_push(in ST s) { stack[stack_ptr++] = s; }
ST stack_pop() { return stack[--stack_ptr]; }
bool stack_empty() { return stack_ptr == 0; }

// The actual ray tracer, based on https://research.nvidia.com/publication/efficient-sparse-voxel-octrees
bool trace(in vec3 ro, in vec3 rd, out vec2 t, out vec3 pos, out int iter, out float size) {
    stack_reset();

    //-- INITIALIZE --//

    int scale = 0;
    size = root_size;
    pos = vec3(0.);
    vec3 tmid;
    vec3 tmax;
    bool can_push = true;
    float d;
    Node parent = decode(nodes[0]);
    t = isect(pos, size, ro, rd, tmid, tmax);
    float h = t.y;
    if (!(t.y >= t.x && t.y >= 0.0)) { return false; }

    // Initial push, sort of
    // If the minimum is before the middle in this axis, we need to go to the first one (-rd)
    vec3 idx = mix(-sign(rd), sign(rd), lessThanEqual(tmid, vec3(t.x)));
    scale = 1;
    size *= 0.5;
    pos += 0.5 * size * idx;

    iter = MAX_ITER;
    while (iter --> 0) { // `(iter--) > 0`; equivalent to `for(int i=128;i>0;i--)`

        t = isect(pos, size, ro, rd, tmid, tmax);

        if (can_push) {
            bool leaf = leaf(parent, idx);
            uint pointer =  parent.pointer[uidx(idx)];

            // We've hit a nonempty leaf voxel, stop now
            if (leaf && pointer != 0)
                return true;

            if (!leaf) {
                //-- PUSH --//

                if (t.y < h) stack_push(ST(parent, pos, scale, idx, h)); // don't add this if we would leave the parent voxel as well

                h = t.y;
                parent = decode(nodes[pointer]);
                scale++;
                size *= 0.5;
                idx = mix(-sign(rd), sign(rd), lessThanEqual(tmid, vec3(t.x)));
                pos += 0.5 * size * idx;
                continue;
            }
        }

        //-- ADVANCE --//

        // Advance for every direction where we're hitting the middle (tmax = tmid)
        vec3 old = idx;
        idx = mix(idx, sign(rd), equal(tmax, vec3(t.y)));
        pos += mix(vec3(0.), sign(rd), notEqual(old, idx)) * size;

        // If idx hasn't changed, we're at the last child in this voxel
        if (idx == old) {
            //return false;

            //-- POP --//

            if (stack_empty() || scale == 0) return false; // We've investigated every voxel on the ray

            ST s = stack_pop();
            parent = s.parent;
            pos = s.pos;
            scale = s.scale;
            size = root_size * exp2(float(-scale));
			idx = s.idx;
            h = s.h;

            can_push = false; // No push-pop inf loops
        } else can_push = true; // We moved, we're good
    }

    return false;
}
