#version 330 core

// This is a Shadertoy emulator, it works pretty well.

uniform float iTime;
uniform vec2 iResolution;
uniform vec4 iMouse;

in vec3 vertColor;

out vec4 fragColor;


void mainImage( out vec4 fragColor, in vec2 fragCoord );

void main() {
    mainImage(fragColor, gl_FragCoord.xy);
    fragColor = pow(fragColor, vec4(2.2)); // mon2lin
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
#define CAMERA_MOVEMENT 0

// Enable this to see the exact raymarched model
//#define MARCH

// The size of the scene. Don't change unless you change the distance function
const float root_size = 4.;
// The resolution, in octree levels. Feel free to play around with this one
const int levels = 9;

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
    vec3 pos;
	int scale; // size = root_size * exp2(float(-scale));
    vec3 idx;
} stack[levels];

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
        d = dist(pos);

        if (d < size*0.5) { // Voxel exists
            //return true;
            if (scale >= levels)// || d < -size)
                return true; // Filled leaf

            if (can_push) {

                //-- PUSH --//

                stack_push(ST(pos, scale, idx)); // TODO don't add this if we would leave
                								// the parent voxel as well (h value)
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
            //return true;

            //-- POP --//

            if (stack_empty() || scale == 0) return false; // We've investigated every voxel on the ray

            ST s = stack_pop();
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


// We want to shade w/ the mipmap
vec3 sky(vec3 dir) {
    return vec3(0.2, 0.2, 1.0);//texture(iChannel1, dir).xyz;
}

// And see the sharp version
vec3 sky_cam(vec3 dir) {
    return vec3(0.2, 0.2, 1.0);//return texture(iChannel0, dir).xyz;
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

    #else

    // The largest component of the vector from the center to the point on the surface,
    //	is necessarily the normal.
    vec3 n = (ro+rd*t.x - pos);
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
            vec3 c = sky(d);//*1.8;//vec3(2.);
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
    vec3 lightDir = /*normalize(vec3(0.2,1.,0.1));/*/reflect(rd,n);
    return 2.*brdf(lightDir, -rd, n, mat, (float(iter)/float(MAX_ITER)));
    #endif
}


void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 uv = fragCoord / iResolution.xy;
    uv *= 2.;
    uv -= 1.;
    uv.x *= iResolution.x / iResolution.y;

    #if CAMERA_MOVEMENT
    float r = iTime;
    #else
    float r = 12.*iMouse.x/iResolution.x;
    #endif
    vec3 ro = vec3(2.*sin(0.5*r),1.5-iMouse.y/iResolution.y,1.6*cos(0.5*r));
    //ro = vec3(0.);
    vec3 lookAt = vec3(0.);//1.,sin(iTime)*1.,cos(iTime)*0.8);
    vec3 cameraDir = normalize(lookAt-ro);//vec3(0.,-1.,1.));
    vec3 up = vec3(0.,1.,0.);
    vec3 left = normalize(cross(cameraDir, up)); // Might be right
    vec3 rd = cameraDir;
    float FOV = 0.5; // Not actual FOV, just a multiplier
    rd += FOV * up * uv.y;
    rd += FOV * left * uv.x;
    rd = normalize(rd);

    vec2 t;
    vec3 pos;
    float size;
    int iter;

    vec3 col = trace(ro, rd, t, pos, iter, size) ? shade(ro, rd, t, iter, pos) : vec3(iter)/float(MAX_ITER);//sky_cam(rd);

    fragColor = vec4(col,1.0);
}
