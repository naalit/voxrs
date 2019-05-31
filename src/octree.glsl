#define STACKLESS

struct Node { // 256-bit
     // The last bit in each pointer is the nonleaf mask, so x & 1 is nonleaf and x >> 1 is pointer
     // Also, an empty leaf is all zeroes, so x == 0 checks that
    uint pointer[8];
};

buffer octree_buffer {
    Node tree[];
};

uint u_idx(vec3 idx) {
    return 0u
        | uint(idx.x > 0.0) << 2
        | uint(idx.y > 0.0) << 1
        | uint(idx.z > 0.0);
}

#ifndef STACKLESS
const int MAX_LEVELS = 8;

struct ST {
    Node parent;
    vec3 pos;
    vec3 idx;
    float size;
    float h;
} stack[MAX_LEVELS];

int stack_ptr = 0; // Next open index
void stack_reset() { stack_ptr = 0; }
void stack_push(in ST s) { stack[stack_ptr++] = s; }
ST stack_pop() { return stack[--stack_ptr]; }
bool stack_empty() { return stack_ptr == 0; }
#else
struct ST {
    Node parent;
    uint parent_pointer;
    vec3 pos;
    vec3 idx;
    float size;
    float h;
};
#endif

// `rdi` is 1/rd, assumed to have been precomputed
vec2 isect(in vec3 ro, in vec3 rdi, in vec3 pos, in float size, out vec3 tmid, out vec3 tmax) {
    vec3 mn = pos - 0.5 * size;
    vec3 mx = mn + size;
    vec3 t1 = (mn-ro) * rdi;
    vec3 t2 = (mx-ro) * rdi;
    vec3 tmin = min(t1, t2);
    tmax = max(t1, t2);

    tmid = (pos-ro) * rdi;

    return vec2(max(tmin.x, max(tmin.y, tmin.z)), min(tmax.x, min(tmax.y, tmax.z)));
}

bool trace(in vec3 ro, in vec3 rd, out vec2 t, out int i, out vec3 pos) {
    #ifndef STACKLESS
    stack_reset();
    #endif

    vec3 tstep = sign(rd);
    vec3 rdi = 1.0 / rd; // Inverse for isect

    float root_size = 8.0;
    vec3 root_pos = vec3(2,-2,4);
    pos = root_pos;

    vec3 tmid, tmax;
    t = isect(ro, rdi, pos, root_size, tmid, tmax);
    if (t.x > t.y || t.y <= 0.0) return false;// else return true;
    float h = t.y;

    // If the minimum is before the middle in this axis, we need to go to the first one (-rd)
    //vec3 idx = mix(-tstep, tstep, lessThanEqual(tmid, vec3(t.x)));

    bvec3 q = lessThanEqual(tmid, vec3(t.x));
    vec3 idx = mix(-tstep, tstep, q);
    vec3 tq = mix(tmid, tmax, q); // tmax of the resulting voxel
    idx = mix(-idx, idx, greaterThanEqual(tq, vec3(0))); // Don't worry about voxels behind `ro`
    uint uidx;
    float size = root_size * 0.5;
    pos += 0.5 * size * idx;
    Node parent = tree[0];
    uint parent_pointer = 0;
    bool c = true;
    ST s = ST(parent,parent_pointer,pos,idx,size,h);

    for (i = 0; i < 256; i++) {
        t = isect(ro, rdi, s.pos, s.size, tmid, tmax);

        uidx = u_idx(s.idx);

        uint node = s.parent.pointer[uidx];

        if ((node & 1u) > 0) { // Non-leaf
            if (c) {
                //-- PUSH --//
                #ifndef STACKLESS
                if (t.y < s.h)
                    stack_push(s);
                #endif
                s.h = t.y;
                s.parent_pointer += node >> 1;
                s.parent = tree[s.parent_pointer];
                s.size *= 0.5;
                q = lessThanEqual(tmid, vec3(t.x));
                s.idx = mix(-tstep, tstep, q);
                tq = mix(tmid, tmax, q); // tmax of the resulting voxel
                s.idx = mix(-s.idx, s.idx, greaterThanEqual(tq, vec3(0))); // Don't worry about voxels behind `ro`
                s.pos += 0.5 * s.size * s.idx;
                continue;
            }
        } else if (node != 0) { // Nonempty, but leaf
            pos = s.pos;
            return true;
        }

        //-- ADVANCE --//

        // Advance for every direction where we're hitting the side
        vec3 old = s.idx;
        s.idx = mix(s.idx, tstep, equal(tmax, vec3(t.y)));
        s.pos += mix(vec3(0.0), tstep, notEqual(old, s.idx)) * s.size;

        if (old == s.idx) { // We're at the last child
            //-- POP --//
            //continue;
            // return true;
            #ifdef STACKLESS

            vec3 target = s.pos;
            s.size = root_size;
            s.pos = root_pos;

            t = isect(ro,rdi,s.pos,s.size,tmid,tmax);
            if (t.y <= s.h)
                return false;

            s.parent = tree[0];
            s.parent_pointer = 0;
            float nh = t.y;
            for (int j = 0; j < 100; j++) { // J is there just in case
                s.size *= 0.5;
                s.idx = sign(target-s.pos+0.0001); // Add 0.0001 to avoid zeros
                s.pos += s.idx * s.size * 0.5;
                t = isect(ro, rdi, s.pos, s.size, tmid, tmax);

                // We have more nodes to traverse within this one
                if (t.y > s.h) {
                    uidx = u_idx(s.idx);
                    node = s.parent.pointer[uidx];
                    s.parent_pointer += node >> 1;
                    s.parent = tree[s.parent_pointer];
                    nh = t.y;
                } else break;
            }
            s.h = nh;

            #else
            if (stack_empty()) return false;

            s = stack_pop();
            #endif

            c = false;
            continue;
        }
        c = true;

    }

    return false;
}
