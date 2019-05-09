#version 450

// This is a Shadertoy emulator, it works pretty well.

uniform float iTime;
uniform vec2 iResolution;
uniform vec2 iMouse;
uniform vec3 cameraPos;
uniform vec3 cameraDir;
uniform vec3 cameraUp;

#define OCTREE 0
#define CHUNK_SIZE 16

// Switch between fb39ca's DDA (https://www.shadertoy.com/view/4dX3zl) and Amanatides and Woo's algorithm
#define DDA

struct MatData {
    vec3 color;
    float roughness;
};

buffer mat_list {
    MatData materials[];
};

#if OCTREE
// We'll need to change the max later
#define MAX_LEVELS 8
uniform int levels;

buffer octree {
    dvec4 nodes[];
};
#else

uniform usampler3D chunks;
uniform usampler3D blocks;
uniform vec3 chunk_origin;

#endif
// in vec3 vertColor;

out vec4 fragColor;


void mainImage( out vec4 fragColor, in vec2 fragCoord );

void main() {
    mainImage(fragColor, gl_FragCoord.xy);
    fragColor = pow(fragColor, vec4(2.2)); // mon2lin
}


// -------------------------------------------

#if OCTREE

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

#endif

// -------------------------------------------


// Sample the envmap in multiple places and pick the highest valued one. Not really physically accurate if not 1
#define SKY_SAMPLES 1
// How many directions to sample the lighting & BRDF at
// Setting it to 0 disables the envmap and switches to a constant light
#define MAT_SAMPLES 0

// Set this to 1 for a liquid-like animation
#define ANIMATE 0
// Try turning that on and this off to see the animation more clearly
// #define CAMERA_MOVEMENT 0

// Enable this to see the exact raymarched model
// #define MARCH

#ifdef OCTREE
// The size of the scene. Don't change unless you change the distance function
const float root_size = 16.*2.*2.;
#endif

// The maximum iterations for voxel traversal
const int MAX_ITER = 512;
// -----------------------------------------------------------------------------------------


#define PI 3.1415926535
const float IPI = 1./PI;
const float R2PI = sqrt(2./PI);

float sqr(float x) { return x*x; }
#define saturate(x) clamp(x,0.,1.)

vec2 isect(in vec3 pos, in float size, in vec3 ro, in vec3 rd, out vec3 tmid, out vec3 tmax) {
    vec3 mn = pos - 0.5 * size;
    vec3 mx = mn + size;
    vec3 t1 = (mn-ro) / rd;
    vec3 t2 = (mx-ro) / rd;
    vec3 tmin = min(t1, t2);
    tmax = max(t1, t2);
    tmid = (pos-ro)/rd; // tmax;
    return vec2(max(tmin.x, max(tmin.y, tmin.z)), min(tmax.x, min(tmax.y, tmax.z)));
}

#if OCTREE
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
#endif

// -----------------------------------------------------------------------------------------

#if OCTREE
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

#else

const int LOG2_CHUNK = int(log2(CHUNK_SIZE));
int scene_size;

uint getVoxel(ivec3 pos) {
    // return uint(texelFetch(chunks, ivec3(3,1,2), 0).zyx == uvec3(3*16, 1*16, 2*16));
    if (any(lessThan(pos,ivec3(0))) || any(greaterThanEqual(pos,ivec3(scene_size))))
        return 121212; // Bigger than the max for u16, so it will never be this for real
    // return floatBitsToUint(texelFetch(blocks, pos, 0));
    ivec3 chunk = /*pos >> LOG2_CHUNK; /*/ pos / CHUNK_SIZE;
    ivec3 in_chunk = /*pos & (CHUNK_SIZE - 1); /*/ pos % CHUNK_SIZE;
    uvec3 offset = texelFetch(chunks, chunk, 0).zyx;
    // return uint(ivec3(offset) == chunk*CHUNK_SIZE);
    // return uint((ivec3(offset) - chunk*CHUNK_SIZE).x < -32);
    return texelFetch(blocks, ivec3(offset)*CHUNK_SIZE + in_chunk, 0);
}

// Regular grid
uint trace(in vec3 ro, in vec3 rd, out vec2 t, out vec3 pos, out int iter, out float size, out vec3 normal) {
    size = 1.0;

    ivec3 total_size = textureSize(blocks, 0);
    scene_size = total_size.x;
    float root_size = float(total_size.x) * 0.5;

    // pos = chunk_origin + root_size;//vec3(root_size); // Center
    vec3 tmid,
        // tmax_root,
        tmax;
    // We translate it to "grid space" first, so pos = vec3(0) is actually chunk_origin in world coordinates
    vec3 ro_chunk = ro - chunk_origin + root_size;
    // t = isect(pos, root_size*2.0, ro_chunk, rd, tmid, tmax_root);
    pos = ro_chunk;// + (t.x+0.01) * rd;
    ivec3 ipos = ivec3(floor(pos));
    ivec3 istep = ivec3(sign(rd));
    isect(vec3(ipos) + 0.5, size, ro_chunk, rd, tmid, tmax);
    vec3 tdelta = abs(1.0 / rd);
    vec3 sideDist = (sign(rd) * (vec3(ipos) - ro_chunk) + (sign(rd) * 0.5) + 0.5) * tdelta;
    bvec3 mask;

    iter = MAX_ITER;
    while (iter --> 0) {
        uint voxel =
            getVoxel(ipos);
            // texelFetch(chunks, ipos, 0).r;
        if (voxel == 121212)
            return 0;
        if (voxel != 0) {
            t = isect(vec3(ipos) + 0.5, size, ro_chunk, rd, tmid, tmax);
            normal = vec3(mask);
            pos = vec3(ipos) + 0.5 + chunk_origin - root_size; // Translate it back to world space
            return voxel;
        }

        mask = lessThanEqual(sideDist.xyz, min(sideDist.yzx, sideDist.zxy));
        sideDist += vec3(mask) * tdelta;
        ipos += ivec3(mask) * istep;
    }
    return 0;
}


#endif

// -----------------------------------------------------------------------------------------

#include <sky.glsl>

// -------------------------------------------------------


// And see the sharp version
vec3 sky_cam(vec3 pos, vec3 dir) {
    return sky(pos,dir);//vec3(0.2, 0.2, 1.0);//return texture(iChannel0, dir).xyz;
}

// https://shaderjvo.blogspot.com/2011/08/van-ouwerkerks-rewrite-of-oren-nayar.html
vec3 oren_nayar(vec3 from, vec3 to, vec3 normal, MatData mat) {
    // Roughness, A and B
    float roughness = mat.roughness;
    float roughness2 = roughness * roughness;
    vec2 oren_nayar_fraction = roughness2 / (roughness2 + vec2(0.33, 0.09));
    vec2 oren_nayar = vec2(1, 0) + vec2(-0.5, 0.45) * oren_nayar_fraction;
    // Theta and phi
    vec2 cos_theta = saturate(vec2(dot(normal, from), dot(normal, to)));
    vec2 cos_theta2 = cos_theta * cos_theta;
    float sin_theta = sqrt((1.-cos_theta2.x) * (1.-cos_theta2.y));
    vec3 light_plane = normalize(from - cos_theta.x * normal);
    vec3 view_plane = normalize(to - cos_theta.y * normal);
    float cos_phi = saturate(dot(light_plane, view_plane));
    // Composition
    float diffuse_oren_nayar = cos_phi * sin_theta / max(cos_theta.x, cos_theta.y);
    float diffuse = cos_theta.x * (oren_nayar.x + oren_nayar.y * diffuse_oren_nayar);

    return mat.color * diffuse;
}


// These bits from https://simonstechblog.blogspot.com/2011/12/microfacet-brdf.html

float schlick_g1(vec3 v, vec3 n, float k) {
    float ndotv = dot(n, v);
    return ndotv / (ndotv * (1. - k) + k);
}

vec3 brdf(vec3 from, vec3 to, vec3 n, MatData mat) {
    float ior = 1.5;

    // Half vector
    vec3 h = normalize(from + to);

    // Schlick fresnel
    float f0 = (1.-ior)/(1.+ior);
    f0 *= f0;
    float fresnel = f0 + (1.-f0)*pow(1.-dot(from, h), 5.);

    // Beckmann microfacet distribution
    float m2 = sqr(mat.roughness);
    float nh2 = sqr(saturate(dot(n,h)));
    float dist = (exp( (nh2 - 1.)
    	/ (m2 * nh2)
    	))
        / (PI * m2 * nh2*nh2);

    // Smith's shadowing function with Schlick G1
    float k = mat.roughness * R2PI;
    float geometry = schlick_g1(from, n, k) * schlick_g1(to, n, k);

    return saturate((fresnel*geometry*dist)/(4.*dot(n, from)*dot(n, to))
        + (1.-f0)*oren_nayar(from, to, n, mat));
}


// -----------------------------------------------------------------------------------------


// By Dave_Hoskins https://www.shadertoy.com/view/4djSRW
vec3 hash33(vec3 p3)
{
	p3 = fract(p3 * vec3(.1031, .1030, .0973));
    p3 += dot(p3, p3.yxz+19.19);
    return fract((p3.xxy + p3.yxx)*p3.zyx);

}

vec3 roundN(vec3 x, float n) {
    return round(x*n)/n;
}

float voxel(vec3 pos) {
    ivec3 cpos = ivec3(pos - chunk_origin + float(scene_size)*0.5);
    return float(getVoxel(cpos));
}

// From gltracy - https://www.shadertoy.com/view/MdBGRm
void occlusion( vec3 v, vec3 n, out vec4 side, out vec4 corner ) {
	vec3 s = n.yzx;
	vec3 t = n.zxy;

	side = vec4 (
		voxel( v - s ),
		voxel( v + s ),
		voxel( v - t ),
		voxel( v + t )
	);

	corner = vec4 (
		voxel( v - s - t ),
		voxel( v + s - t ),
		voxel( v - s + t ),
		voxel( v + s + t )
	);
}

float filterf( vec4 side, vec4 corner, vec2 tc ) {
	vec4 v = side.xyxy + side.zzww + corner;

	return mix( mix( v.x, v.y, tc.y ), mix( v.z, v.w, tc.y ), tc.x ) * 0.25;
}

float ao( vec3 v, vec3 n, vec2 tc ) {
	vec4 side, corner;

	occlusion( v + n, abs( n ), side, corner );

	return 1.0 - filterf( side, corner, tc );
}

MatData mat_lookup(uint mat) {
    return materials[mat];
    /*
    switch(mat)
    {
        case 1:
            return Material(vec3(0.7,0.8,0.8), 0.2);
        case 2:
            return Material(vec3(0.3,0.7,0.5),0.9);
        default:
            return Material(vec3(1.0,1.0,1.0),1.0);
    }
    */
}

vec3 shade(uint m, vec3 ro, vec3 rd, vec2 t, int iter, vec3 pos, vec3 mask) {
    // // The biggest component of intersection_pos - voxel_pos is the normal direction
    // #ifdef MARCH
    // /*
    // // The normal here isn't really accurate, the surface is too high-frequency
    // float e = 0.0001;
    // vec3 eps = vec3(e,0.0,0.0);
	// vec3 n = normalize( vec3(
    //        dist(pos+eps.xyy) - dist(pos-eps.xyy),
    //        dist(pos+eps.yxy) - dist(pos-eps.yxy),
    //        dist(pos+eps.yyx) - dist(pos-eps.yyx) ) );
	// */
    // // This pretends the Mandelbulb is actually a sphere, but it looks okay w/ AO.
    // vec3 n = normalize(pos);
    // // And this isn't accurate even for a sphere, but it ensures the edges are visible.
    // n = faceforward(n,-rd,-n);
    // vec3 p = pos;
    // #else
    //
    // // The largest component of the vector from the center to the point on the surface,
    // //	is necessarily the normal.
    vec3 p = ro+rd*t.x;
    // vec3
    vec3 n = (p - pos);
    n = sign(n) * (abs(n.x) > abs(n.y) ? // Not y
        (abs(n.x) > abs(n.z) ? vec3(1., 0., 0.) : vec3(0., 0., 1.)) :
    	(abs(n.y) > abs(n.z) ? vec3(0., 1., 0.) : vec3(0., 0., 1.)));
    // #endif

    MatData mat = mat_lookup(m);
    // Material(normalize(abs(n)+abs(pos)+vec3(0.5)), 0.9); // Color from normal+position of voxel
    // mat.color = vec3(1.,.9,.7);
    #if MAT_SAMPLES
    vec3 acc = vec3(0.);
    int j;
    for (j = 0; j < MAT_SAMPLES; j++) {
        vec3 lightDir;
        vec3 lightCol = vec3(0.);
        for (int i = 0; i < SKY_SAMPLES; i++) {
            vec3 d = hash33(/*rd+t.x-t.y+float(iter)+*/.2*pos+0.5*n+float(i+j*SKY_SAMPLES));
            //d = reflect(rd,n);
            d = normalize(d);
            //d = faceforward(d, n, -d);//normalize(vec3(0.2,1.,0.3));
            vec3 c = sky(pos,d);//*1.8;//vec3(2.);
            if (length(c) > length(lightCol)) {
                lightCol = c;
                lightDir = d;
            }
        }

        acc +=
            ao(pos,n,t) +
            2.*pow(lightCol, vec3(2.2)) * brdf(lightDir, -rd, n, mat);
    }
    return acc / float(j);
	#else
    vec3 lightDir = major_dir();//reflect(rd,n);
    vec3 c = sky(p,lightDir);
    vec2 t_;
    vec3 pos_;
    int iter_ = MAX_ITER;
    float size_;
    bool shadow = false;//trace(p+1.1*n*(root_size/exp2(levels)), vec3(0.0,1.0,0.0), t_, pos_, iter_, size_);
    vec2 tc =
        ( fract( p.yz ) * mask.x ) +
        ( fract( p.zx ) * mask.y ) +
        ( fract( p.xy ) * mask.z );
    return (shadow ? vec3(0.3) :
        (0.1+ao(pos,n,tc)*0.2)*mat.color + c*brdf(-lightDir, -rd, n, mat));
    #endif
}


void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    /*Node a = decode(nodes[0]);
    bool x = a == Node(
        bool[](true, true, false, true, true, true, true, true),
        uint[](0, 0, 1, 0, 1, 0, 0, 0)
    );
    int i = 3;*/
    /*Node b = decode(nodes[1]);
    //bool x = true;
    bool x = b == Node(
        bool[](true,true,true,true,true,true,true,true),
        uint[](0, 0, 0, 1, 0, 0, 0, 0)
    );

    fragColor = vec4(vec3(x), 1);
    return;*/

    vec2 uv = fragCoord / iResolution.xy;
    uv *= 2.;
    uv -= 1.;
    uv.y *= iResolution.y / iResolution.x;

    /*
    #if CAMERA_MOVEMENT
    float r = iTime;
    #else
    float r = 12.*iMouse.x/iResolution.x;
    #endif
    vec3 ro = vec3(2.*sin(0.5*r),1.5-3.0*iMouse.y/iResolution.y,1.6*cos(0.5*r));
    */
    vec3 ro = cameraPos;
    //ro = vec3(0.);
    //vec3 lookAt = vec3(0.);//1.,sin(iTime)*1.,cos(iTime)*0.8);
    //vec3 cameraDir = normalize(lookAt-ro);//vec3(0.,-1.,1.));
    vec3 up = cameraUp; //vec3(0.,1.,0.);
    vec3 right = normalize(cross(cameraDir, cameraUp)); // Might be right
    vec3 rd = cameraDir;
    float FOV = 1.0; // Not actual FOV, just a multiplier
    rd += FOV * up * uv.y;
    rd += FOV * right * uv.x;
    rd = normalize(rd);

    vec2 t;
    vec3 pos;
    float size;
    int iter;
    vec3 n;

    uint mat = trace(ro, rd, t, pos, iter, size, n);
    vec3 col = mat != 0 ? shade(mat, ro, rd, t, iter, pos, n) : sky_cam(ro, -rd);
    // col = vec3(1.0) * float(iter)/float(MAX_ITER);

    fragColor = vec4(col,1.0);
}
