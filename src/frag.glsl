#version 450

// This is a Shadertoy emulator, it works pretty well.

uniform float iTime;
uniform vec2 iResolution;
uniform vec2 iMouse;
uniform vec3 cameraPos;
uniform vec3 cameraDir;
uniform vec3 cameraUp;

// We'll need to change the max later
#define MAX_LEVELS 8
uniform int levels;

buffer octree {
    dvec4 nodes[];
};

// in vec3 vertColor;

out vec4 fragColor;


void mainImage( out vec4 fragColor, in vec2 fragCoord );

void main() {
    mainImage(fragColor, gl_FragCoord.xy);
    fragColor = pow(fragColor, vec4(2.2)); // mon2lin
}


// -------------------------------------------


struct Node {
    bool[8] leaf;
    uint[8] pointer;
};

// The equivalent function in Rust follows this one closely to allow for testing of this one,
//  so only update both at once
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

// The size of the scene. Don't change unless you change the distance function
const float root_size = 4.;
// The resolution, in octree levels. Feel free to play around with this one
// const int levels = 9;

// The maximum iterations for voxel traversal. Increase if you see raymarching-like
//	hole artifacts at edges, decrease if it's too slow
const int MAX_ITER = 256;
// Note, though, that the fake AO might look weird if you change it much

// These are parameters for the Mandelbulb, change if you want. Higher is usually slower
const float Power = 4.;
const float Bailout = 1.5;
const int Iterations = 6;


// -----------------------------------------------------------------------------------------


// This is from http://blog.hvidtfeldts.net/index.php/2011/09/distance-estimated-3d-fractals-v-the-mandelbulb-different-de-approximations/
// 	because it had code that I could just copy and paste
float dist(vec3 pos) {
    // This function takes ~all of the rendering time, the trigonometry is super expensive
    // So if there are any faster approximations, they should definitely be used
	vec3 z = pos;
	float dr = 1.0;
	float r = 0.0;
    for (int i = 0; i < Iterations; i++) {
		r = length(z);
		if (r>Bailout) break;

		// convert to polar coordinates
		float theta = acos(z.z/r);
        #if ANIMATE
        theta += iTime*0.5;
        #endif
		float phi = atan(z.y,z.x);
        #if ANIMATE
        phi += iTime*0.5;
        #endif
		dr = pow( r, Power-1.0)*Power*dr + 1.0;

		// scale and rotate the point
		float zr = pow( r,Power);
		theta = theta*Power;
		phi = phi*Power;

		// convert back to cartesian coordinates
		z = zr*vec3(sin(theta)*cos(phi), sin(phi)*sin(theta), cos(theta));
        z+=pos;
	}
	return 0.5*log(r)*r/dr;
}


// -----------------------------------------------------------------------------------------


#define PI 3.1415926535
const float IPI = 1./PI;
const float R2PI = sqrt(2./PI);

struct Material {
    vec3 base_color;
    float roughness;
};

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

struct ST {
    Node parent;
    vec3 pos;
	int scale; // size = root_size * exp2(float(-scale));
    vec3 idx;
} stack[MAX_LEVELS];

int stack_ptr = 0; // Next open index
void stack_reset() { stack_ptr = 0; }
void stack_push(in ST s) { stack[stack_ptr++] = s; }
ST stack_pop() { return stack[--stack_ptr]; }
bool stack_empty() { return stack_ptr == 0; }


// -----------------------------------------------------------------------------------------

#ifndef MARCH
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
    //if (!(t.y >= t.x && t.y >= 0.0)) { return false; }

    // Initial push, sort of
    // If the minimum is before the middle in this axis, we need to go to the first one (-rd)
    vec3 idx = mix(-sign(rd), sign(rd), lessThanEqual(tmid, vec3(t.x)));
    scale = 1;
    size *= 0.5;
    pos += 0.5 * size * idx;

    iter = MAX_ITER;
    while (iter --> 0) { // `(iter--) > 0`; equivalent to `for(int i=128;i>0;i--)`
        //if (iter == 1) return true;
        t = isect(pos, size, ro, rd, tmid, tmax);

        //t = idx.xy*8.;
        //return true;

    	/*if (t.x > t.y || t.y < 0.0)
            return false; // We missed it
        */
        // d = dist(pos);

        // if (d < size*0.5) { // Voxel exists

        bool leaf = leaf(parent, idx);
        uint pointer =  parent.pointer[uidx(idx)];

        // We've hit a nonempty leaf voxel, stop now
        if (leaf && pointer != 0)
            return true;

        if (!leaf) {
            // We're not going farther, we've hit the maximum level.
            // Really, this should never happen with a well constructed octree...
            // if (scale >= levels)
            //     return true;

            if (can_push) {
                //-- PUSH --//

                stack_push(ST(parent, pos, scale, idx)); // TODO don't add this if we would leave the parent voxel as well (h value)

                parent = voxel(parent, idx);
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

            can_push = false; // No push-pop inf loops
        } else can_push = true; // We moved, we're good
    }

    return false;
}

#else
// Simple ray marcher for visualizing the exact distance function
bool trace(in vec3 ro, in vec3 rd, out vec2 t, out vec3 pos, out int iter, out float size) {
	size = 0.;

    t = vec2(0.);
    pos = ro;
    iter = MAX_ITER;
    while (iter --> 0 && t.x < root_size) {
        float d = dist(pos);
        if (d < 0.01)
            return true;
        t += d;
        pos += rd*d;
    }
    return false;
}
#endif

// -----------------------------------------------------------------------------------------


// How fast time goes, in days/second. So 0.00001157407 is real time, 0.00069 is 1 hour per minute
#define SUN_SPEED 0.0069
// 1.0 is default, goes both ways
#define SUN_SIZE 1.0
// Makes the sun & glow dissapear below the horizon
#define STRICT_HORIZ 0

#define STAR_DENSITY 0.3

//#define PI 3.1415926535
const float TAU = PI * 2.0;

// DERIVATION OF RAY - Y-PLANE INTERSECTION
// p.y = ro.y+t*rd.y;
// y = ro.y+t*rd.y
// y-ro.y = t*rd.y
// (y-ro.y)/rd.y = t

// from https://www.shadertoy.com/view/4djSRW
float hash(vec3 p)
{
	vec3 p3  = fract(p * .1031);
    p3 += dot(p3, p3.yzx + 19.19);
    return fract((p3.x + p3.y) * p3.z);
}

// from https://www.shadertoy.com/view/4sfGzS
float noise3(vec3 x) {
    vec3 p = floor(x);
    vec3 f = fract(x);
    f = f*f*(3.0-2.0*f);

    return mix(mix(mix( hash(p+vec3(0,0,0)),
                        hash(p+vec3(1,0,0)),f.x),
                   mix( hash(p+vec3(0,1,0)),
                        hash(p+vec3(1,1,0)),f.x),f.y),
               mix(mix( hash(p+vec3(0,0,1)),
                        hash(p+vec3(1,0,1)),f.x),
                   mix( hash(p+vec3(0,1,1)),
                        hash(p+vec3(1,1,1)),f.x),f.y),f.z);
}

const mat3 m = mat3( 0.00,  0.80,  0.60,
                    -0.80,  0.36, -0.48,
                    -0.60, -0.48,  0.64 );

float noise(in vec2 pos) {
    vec3 q = 8.0*pos.xyx;
    float f  = 0.5000*noise3( q ); q = m*q*2.01;
    f += 0.2500*noise3( q ); q = m*q*2.02;
    f += 0.1250*noise3( q ); q = m*q*2.03;
    f += 0.0625*noise3( q ); q = m*q*2.01;
    return f;
}
float noise(in vec3 pos) {
    vec3 q = (8.0*pos);//.xyx;
    float f  = 0.5000*noise3( q ); q = m*q*2.01;
    f += 0.2500*noise3( q ); q = m*q*2.02;
    f += 0.1250*noise3( q ); q = m*q*2.03;
    f += 0.0625*noise3( q ); q = m*q*2.01;
    return f;
}
/*
vec3 roundN(vec3 x, float n) {
    return round(x*n)/n;
}
*/

// Whichever is brighter, the sun or the moon
vec3 major_dir() {
    float sun_speed = SUN_SPEED*TAU;
    vec3 sun_dir = normalize(vec3(sin(iTime*sun_speed), cos(iTime*sun_speed), 0.0)); //vec2(float(iFrame), 0.);
    float sun_height = dot(sun_dir, vec3(0.0,-1.0,0.0));

    float moon_phase = cos(iTime*sun_speed/(TAU*29.5));
    float moon_offset = PI*moon_phase;
    vec3 moon_dir = normalize(vec3(sin(iTime*sun_speed+moon_offset), cos(iTime*sun_speed+moon_offset), 0.0));
    return mix(moon_dir, sun_dir, smoothstep(0.0,0.1,sun_height));
}

vec3 sky(vec3 ro, vec3 rd) {
    float sky_y = 80.0;
    float horiz = dot(rd,vec3(0.0,-1.0,0.0));
    #if STRICT_HORIZ
    if (horiz <= 0.0)
        return vec3(0.0);
    #endif
    float t = (sky_y - ro.y) / rd.y;
    vec2 uv = (ro + t*rd).xz;

    vec3 sky_color = vec3(0.3,0.55,0.85);
    vec3 sun_color = vec3(30.0,16.0,6.0)*1.1;

    float sun_size = pow(0.985,SUN_SIZE);
    float sun_speed = SUN_SPEED*TAU;
    vec3 sun_dir = normalize(vec3(sin(iTime*sun_speed), cos(iTime*sun_speed), 0.0)); //vec2(float(iFrame), 0.);
    float sun_height = dot(sun_dir, vec3(0.0,-1.0,0.0));
    //sun_color = pow(sun_color/20.0,vec3(1.0+48.0*sun_height))*20.0;
    //sun_size *= (20.0-sun_height)/20.0;

    float sun_d = acos(dot(rd, sun_dir));
    float sun = smoothstep(TAU*sun_size, TAU, TAU-sun_d);

    vec3 stars = vec3(smoothstep(0.8-STAR_DENSITY*0.1,0.8,noise(min(vec3(sin(iTime*sun_speed*2.0), cos(iTime*sun_speed*2.0), 0.0)+rd*20.0,20.0))));
    vec3 sky_col = sign(horiz)*mix(vec3(0.0), sky_color, smoothstep(-0.6,-0.4,sun_height));
    stars = sign(horiz)*mix(stars, vec3(0.0), smoothstep(-0.6,-0.4,sun_height));
    sky_col *= mix(1.2,0.6,horiz);

    vec3 col = mix(sky_col, sun_color, sun);

    vec3 glow = (1.-sun_height)*pow(sun_color * 0.04 * pow(max(0.0, TAU - sun_d)/TAU,10.0), vec3(2.2-sun_height));
    glow += pow((1.-abs(sun_height)-sun_d*0.15) * smoothstep(0.7,1.0,1.0-horiz) * sun_color * 0.03,vec3(2.2-sun_height));

    float cloud_size = 0.4; // [0.0,inf] but 0.0 is no clouds
    float cloud_e = noise((sun_speed*80.0*iTime*4.0+uv)*0.01*(1.0/(10.0*cloud_size)));
    float cloud = smoothstep(0.4,1.0,cloud_e);
    float cloud_fac = cloud * max(0.0,smoothstep(0.0,0.3,horiz));
    col = mix(clamp(col,0.0,1.0),vec3(1.0),cloud_fac);
    glow *= 1.0+(1.0-sun_height)*4.0*smoothstep(0.1,0.75,cloud_e)*max(0.0,horiz);

    col += mix(clamp(glow,0.0,1.0),vec3(0.0),cloud_fac);

    // 0=new moon, 1=full moon
    float moon_phase = cos(iTime*sun_speed/(TAU*29.5));
    float moon_offset = PI*moon_phase;
    vec3 moon_dir = normalize(vec3(sin(iTime*sun_speed+moon_offset), cos(iTime*sun_speed+moon_offset), 0.0));

    float moon_d = acos(dot(rd, moon_dir));
    float moon_s = smoothstep(TAU-0.06, TAU-0.05, TAU-moon_d);

    // SPHERE NORMAL, WITHOUT A SPHERE!
    // 	the normal should be -moon_dir at the center,
    //	and cross(moon_dir,cross(moon_dir,rd)) at an edge
    vec3 moon_n = normalize(mix(cross(moon_dir, cross(moon_dir,rd)), -moon_dir, smoothstep(TAU-0.06,TAU,TAU-moon_d)));
    // We use the normal cosine-falloff law to generate realistic moon phases
    //	In reality, though, the moon should be offset from the sun sideways too, and the direction
    //	from the sun to moon isn't exactly the same as the direction from the sun to the earth.
    float moon = max(0.6,1.0-smoothstep(0.5,1.0,noise(3.0+5.0*(rd-moon_dir))))*dot(moon_n,-sun_dir);
    vec3 moon_c = 0.8*max(mix(stars, vec3(moon),moon_s),0.0);//*max(0.3,1.0-length(col)));
    col = col*0.2+max(col*0.8,moon_c);
    return col;
}


// -------------------------------------------------------


// And see the sharp version
vec3 sky_cam(vec3 pos, vec3 dir) {
    return sky(pos,dir);//vec3(0.2, 0.2, 1.0);//return texture(iChannel0, dir).xyz;
}

// https://shaderjvo.blogspot.com/2011/08/van-ouwerkerks-rewrite-of-oren-nayar.html
vec3 oren_nayar(vec3 from, vec3 to, vec3 normal, Material mat) {
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

    return mat.base_color * diffuse;
}


// These bits from https://simonstechblog.blogspot.com/2011/12/microfacet-brdf.html

float schlick_g1(vec3 v, vec3 n, float k) {
    float ndotv = dot(n, v);
    return ndotv / (ndotv * (1. - k) + k);
}

vec3 brdf(vec3 from, vec3 to, vec3 n, Material mat, float ao) {
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
        + ao*(1.-f0)*oren_nayar(from, to, n, mat));
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

vec3 shade(vec3 ro, vec3 rd, vec2 t, int iter, vec3 pos) {
    // The biggest component of intersection_pos - voxel_pos is the normal direction
    #ifdef MARCH
    /*
    // The normal here isn't really accurate, the surface is too high-frequency
    float e = 0.0001;
    vec3 eps = vec3(e,0.0,0.0);
	vec3 n = normalize( vec3(
           dist(pos+eps.xyy) - dist(pos-eps.xyy),
           dist(pos+eps.yxy) - dist(pos-eps.yxy),
           dist(pos+eps.yyx) - dist(pos-eps.yyx) ) );
	*/
    // This pretends the Mandelbulb is actually a sphere, but it looks okay w/ AO.
    vec3 n = normalize(pos);
    // And this isn't accurate even for a sphere, but it ensures the edges are visible.
    n = faceforward(n,-rd,-n);
    vec3 p = pos;
    #else

    // The largest component of the vector from the center to the point on the surface,
    //	is necessarily the normal.
    vec3 p = ro+rd*t.x;
    vec3 n = (p - pos);
    n = sign(n) * (abs(n.x) > abs(n.y) ? // Not y
        (abs(n.x) > abs(n.z) ? vec3(1., 0., 0.) : vec3(0., 0., 1.)) :
    	(abs(n.y) > abs(n.z) ? vec3(0., 1., 0.) : vec3(0., 0., 1.)));
    #endif

    Material mat = Material(normalize(abs(n)+abs(pos)+vec3(0.5)), 0.9); // Color from normal+position of voxel
    mat.base_color = vec3(1.,.9,.7);
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
            //0.05*(float(iter)/float(MAX_ITER))*mat.base_color // Fake AO - try commenting out the next line
            2.*pow(lightCol, vec3(2.2)) * brdf(lightDir, -rd, n, mat, (float(iter)/float(MAX_ITER)));
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
    return (shadow ? vec3(0.3) : vec3(float(iter_)/float(MAX_ITER))) * c*brdf(-lightDir, -rd, n, mat, (float(iter)/float(MAX_ITER)));
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
    float FOV = 0.5; // Not actual FOV, just a multiplier
    rd += FOV * up * uv.y;
    rd += FOV * right * uv.x;
    rd = normalize(rd);

    vec2 t;
    vec3 pos;
    float size;
    int iter;

    vec3 col = trace(ro, rd, t, pos, iter, size) ? shade(ro, rd, t, iter, pos) : sky_cam(ro, -rd);

    fragColor = vec4(col,1.0);
}
