#define PBR_SKY
// How fast time goes, in days/second. So 0.00001157407 is real time, 0.00069 is 1 hour per minute
#define SUN_SPEED 0.00069
// How fast the moon phases go. 1.0 is a normal moon phase - 29.5 days/cycle
#define MOON_SPEED 1.0
// Makes the moon, sun, & glow dissapear below the horizon
#define STRICT_HORIZ 0
// Uses 3-octave fBm for the clouds, instead of regular 1-octave noise. I can't decide which I like better, so try both
#define FBM_CLOUDS 1

#define CLOUD_DENSITY 0.45
#define STAR_DENSITY 0.5

#define PI 3.1415926535
const float TAU = PI * 2.0;

// from https://www.shadertoy.com/view/4djSRW
float hash(vec3 p)
{
	vec3 p3  = fract(p * .1031);
    p3 += dot(p3, p3.yzx + 19.19);
    return fract((p3.x + p3.y) * p3.z);
}

// from https://www.shadertoy.com/view/4sfGzS
float noise3( in vec3 x )
{
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
float fBm(in vec3 pos) {
    return noise(pos) + (noise(pos*1.99) - 0.5)/1.99 + (noise(pos*3.98) - 0.5)/3.98;
}
float fBm(in vec2 pos) {
    return fBm (pos.xyx);
}

// A bunch of this code is from 16807
// https://www.shadertoy.com/view/3dBSDW
bool isect_sphere(vec3 ro, vec3 rd, float rad, inout float t0, inout float t1)
{
	vec3 rc = -ro;
	float radius2 = rad * rad;
	float tca = dot(rc, rd);
	float d2 = dot(rc, rc) - tca * tca;
	if (d2 > radius2) return false;
	float thc = sqrt(radius2 - d2);
	t0 = tca - thc;
	t1 = tca + thc;

	return true;
}

#ifdef PBR_SKY
// scattering coefficients at sea level (m)
const vec3 betaR = vec3(5.5e-6, 13.0e-6, 22.4e-6); // Rayleigh
const vec3 betaM = vec3(21e-6); // Mie

// scale height (m)
// thickness of the atmosphere if its density were uniform
const float hR = 7994.0; // Rayleigh
const float hM = 1200.0; // Mie

float rayleigh_phase_func(float mu)
{
	return
			3. * (1. + mu*mu)
	/ //------------------------
				(16. * PI);
}

// Henyey-Greenstein phase function factor [-1, 1]
// represents the average cosine of the scattered directions
// 0 is isotropic scattering
// > 1 is forward scattering, < 1 is backwards
const float g = 0.76;
//const float TAU = 2.0 * PI;
float mie_phase(float mu)
{
	return
        (3.0 * (1.0 - g*g) * (1.0 + mu*mu)) /
        (4.0 * TAU * (2.0 + g*g) * pow(1.0 + g*g - 2.0*g*mu, 1.5));
}

float henyey_greenstein_phase_func(float mu)
{
	return
						(1. - g*g)
	/ //---------------------------------------------
		((4. + PI) * pow(1. + g*g - 2.*g*mu, 1.5));
}

// Schlick Phase Function factor
// Pharr and  Humphreys [2004] equivalence to g above
const float k = 1.55*g - 0.55 * (g*g*g);
float schlick_phase_func(float mu)
{
	return
					(1. - k*k)
	/ //-------------------------------------------
		(4. * PI * (1. + k*mu) * (1. + k*mu));
}

const float earth_radius = 6360e3; // (m)
const float atmosphere_radius = 6420e3; // (m)

const float sun_power = 20.0;

const float atmosphere = atmosphere_radius;

const int num_samples = 8;
const int num_samples_light = 8;

float approx_air_column_density_ratio_along_2d_ray_for_curved_world(
    float x_start, // distance along path from closest approach at which we start the raymarch
    float x_stop,  // distance along path from closest approach at which we stop the raymarch
    float z2,      // distance at closest approach, squared
    float r,       // radius of the planet
    float H        // scale height of the planet's atmosphere
){

    // GUIDE TO VARIABLE NAMES:
    //  "x*" distance along the ray from closest approach
    //  "z*" distance from the center of the world at closest approach
    //  "r*" distance ("radius") from the center of the world
    //  "h*" distance ("height") from the surface of the world
    //  "*b" variable at which the slope and intercept of the height approximation is sampled
    //  "*0" variable at which the surface of the world occurs
    //  "*1" variable at which the top of the atmosphere occurs
    //  "*2" the square of a variable
    //  "d*dx" a derivative, a rate of change over distance along the ray

    float a = 0.45;
    float b = 0.45;

    float x0 = sqrt(max(r *r -z2, 0.));
    // if ray is obstructed
    if (x_start < x0 && -x0 < x_stop && z2 < r*r)
    {
        // return ludicrously big number to represent obstruction
        return 1e20;
    }

    float r1      = r + 6.*H;
    float x1      = sqrt(max(r1*r1-z2, 0.));
    float xb      = x0+(x1-x0)*b;
    float rb2     = xb*xb + z2;
    float rb      = sqrt(rb2);
    float d2hdx2  = z2 / sqrt(rb2*rb2*rb2);
    float dhdx    = xb / rb;
    float hb      = rb - r;
    float dx0     = x0          -xb;
    float dx_stop = abs(x_stop )-xb;
    float dx_start= abs(x_start)-xb;
    float h0      = (0.5 * a * d2hdx2 * dx0      + dhdx) * dx0      + hb;
    float h_stop  = (0.5 * a * d2hdx2 * dx_stop  + dhdx) * dx_stop  + hb;
    float h_start = (0.5 * a * d2hdx2 * dx_start + dhdx) * dx_start + hb;

    float rho0  = exp(-h0/H);
    float sigma =
        sign(x_stop ) * max(H/dhdx * (rho0 - exp(-h_stop /H)), 0.)
      - sign(x_start) * max(H/dhdx * (rho0 - exp(-h_start/H)), 0.);

    // NOTE: we clamp the result to prevent the generation of inifinities and nans,
    // which can cause graphical artifacts.
    return min(abs(sigma),1e20);
}

// "approx_air_column_density_ratio_along_3d_ray_for_curved_world" is just a convenience wrapper
//   for the above function that works with 3d vectors.
float approx_air_column_density_ratio_along_3d_ray_for_curved_world (
    vec3  P, // position of viewer
    vec3  V, // direction of viewer (unit vector)
    float x, // distance from the viewer at which we stop the "raymarch"
    float r, // radius of the planet
    float H  // scale height of the planet's atmosphere
){
    float xz = dot(-P,V);           // distance ("radius") from the ray to the center of the world at closest approach, squared
    float z2 = dot( P,P) - xz * xz; // distance from the origin at which closest approach occurs
    return approx_air_column_density_ratio_along_2d_ray_for_curved_world( -xz, x-xz, z2, r, H );
}

bool get_sun_light(
	vec3 ro,
    vec3 rd,
	inout float optical_depthR,
	inout float optical_depthM
){
	float t0, t1;
	isect_sphere(ro, rd, atmosphere, t0, t1);

    // // this is the implementation using classical raymarching
	// float march_pos = 0.;
	// float march_step = t1 / float(num_samples_light);
    //
	// for (int i = 0; i < num_samples_light; i++) {
	// 	vec3 s =
	// 		ray.origin +
	// 		ray.direction * (march_pos + 0.5 * march_step);
	// 	float height = length(s) - earth_radius;
	// 	if (height < 0.)
	// 		return false;
    //
	// 	optical_depthR += exp(-height / hR) * march_step;
	// 	optical_depthM += exp(-height / hM) * march_step;
    //
	// 	march_pos += march_step;
	// }

    // this is the implementation using a fast closed form approximation
    optical_depthR =
        approx_air_column_density_ratio_along_3d_ray_for_curved_world (
            ro,    // position of viewer
            rd, // direction of viewer (unit vector)
            t1, // distance from the viewer at which we stop the "raymarch"
            earth_radius, // radius of the planet
            hR  // scale height of the planet's atmosphere
        );
    optical_depthM =
        approx_air_column_density_ratio_along_3d_ray_for_curved_world (
            ro,    // position of viewer
            rd, // direction of viewer (unit vector)
            t1, // distance from the viewer at which we stop the "raymarch"
            earth_radius, // radius of the planet
            hM  // scale height of the planet's atmosphere
        );

	return true;
}
#endif

vec3 major_dir() {
	float sun_speed = SUN_SPEED * TAU;
	return vec3(sin(iTime * sun_speed), cos(iTime * sun_speed), 0.0);
}

vec3 sky(vec3 ro, vec3 rd)
{
	// return vec3(1.0);
    float horiz = dot(rd,vec3(0.0,1.0,0.0));
    #if 1
    //STRICT_HORIZ
    if (horiz <= 0.0)
        return vec3(0.0);
    #endif

    float sun_speed = SUN_SPEED * TAU;

    // The clouds are on an xz plane at this y-level
    // That means they move and just look visually different from the other things, which use polar coordinates
    float sky_y = 120.0;
    float t = (sky_y - ro.y) / rd.y;
    // This is just the ray's x and z when it intersects the cloud plane
    vec2 uv = (ro + t*rd).xz;

    float cloud_size = 0.4; // [0.0,inf] but 0.0 is no clouds
    // The *320 just speeds it up relative to the sun, because clouds pass more than once per day
    float cloud_e =
        #if FBM_CLOUDS
        fBm
        #else
        noise
        #endif
        ((sun_speed*320.0*iTime+uv)*0.01*(1.0/(10.0*cloud_size)));
    // More discrete clouds and sky
    float cloud = smoothstep(1.0-CLOUD_DENSITY,1.0,cloud_e);
    float cloud_fac = cloud * max(0.0,smoothstep(0.0,0.3,horiz));

    vec3 sun_dir = vec3(sin(iTime * sun_speed), cos(iTime * sun_speed), 0.0);
	float sun_height = dot(sun_dir, vec3(0.0, 1.0, 0.0));


	#ifdef PBR_SKY
    ro += vec3(0.0, earth_radius+1e2, 0.0);
	// "pierce" the atmosphere with the viewing ray
	float t0, t1;
	if (!isect_sphere(
		ro, rd, atmosphere, t0, t1)) {
		return vec3(0);
	}

	float march_step = t1 / float(num_samples);


	// cosine of angle between view and light directions
	float mu = dot(rd, sun_dir);

	// Rayleigh and Mie phase functions
	// A black box indicating how light is interacting with the material
	// Similar to BRDF except
	// * it usually considers a single angle
	//   (the phase angle between 2 directions)
	// * integrates to 1 over the entire sphere of directions
	float phaseR = rayleigh_phase_func(mu);
	float phaseM =
#if 1
		henyey_greenstein_phase_func(mu);
#else
		schlick_phase_func(mu);
#endif

	// optical depth (or "average density")
	// represents the accumulated extinction coefficients
	// along the path, multiplied by the length of that path
	float optical_depthR = 0.;
	float optical_depthM = 0.;

	vec3 sumR = vec3(0);
	vec3 sumM = vec3(0);
	float march_pos = 0.;

	for (int i = 0; i < num_samples; i++) {
		vec3 s =
			ro +
			rd * (march_pos + 0.5 * march_step);
		float height = length(s) - earth_radius;

		// integrate the height scale
		float hr = exp(-height / hR) * march_step;
		float hm = exp(-height / hM) * march_step;
		optical_depthR += hr;
		optical_depthM += hm;

		// gather the sunlight
		float optical_depth_lightR = 0.;
		float optical_depth_lightM = 0.;
		bool overground = get_sun_light(
			s, sun_dir,
			optical_depth_lightR,
			optical_depth_lightM);

		if (overground) {
			vec3 tau =
				betaR * (optical_depthR + optical_depth_lightR) +
				betaM * 1.1 * (optical_depthM + optical_depth_lightM);
			vec3 attenuation = exp(-tau);

			sumR += hr * attenuation;
			sumM += hm * attenuation;
		}

		march_pos += march_step;
	}

	vec3 col =
		sun_power *
		(sumR * phaseR * betaR +
		sumM * phaseM * betaM * (1.0 + cloud_e));
	#else
	float sun_d = dot(sun_dir,rd);
	vec3 sky_col = mix(
        vec3(0.4,0.6,1.0),
        vec3(0.1,0.2,0.7),
        rd.y
        );
	vec3 col = smoothstep(0.0,0.1,sun_height)*sky_col;
	col += max(0.0,pow(sun_d,128.0));
	#endif

    vec3 stars = vec3(smoothstep(0.8-STAR_DENSITY*0.1,0.8,
		// They move the same speed as the sun, because both cases are just the Earth spinning
    	noise(min(sun_dir+rd*20.0,20.0))));
    // We're actually mixing sky_col and stars, but we need to keep them separate for now so the moon can
    //	obscure the stars.
    //vec3 sky_col = sign(horiz)*mix(vec3(0.0), sky_color, smoothstep(-0.6,-0.4,sun_height));
    stars = sign(horiz)*mix(stars, vec3(0.0), smoothstep(0.0,0.1,sun_height));

    // The moon's orbit is offset from the sun's "orbit" by 0-180 degrees, depending on the moon phase
    // Because we do it realistically like this, the moon is sometimes visible during the day, but it looks pretty bad
    // 0=new moon, 0.5=full moon
    float moon_phase = cos(MOON_SPEED*iTime*sun_speed/(TAU*29.5));
    // PI, not TAU, because it's never more than 180 degrees off (180 at the full moon)
    float moon_offset = TAU*moon_phase;
    // Here it follows the same path as the Sun, just offset. That's not realistic, though.
    vec3 moon_dir = normalize(vec3(sin(iTime*sun_speed+moon_offset), cos(iTime*sun_speed+moon_offset), 0.0));

    //float moon_d = acos(dot(rd, moon_dir));

    float mu_moon = dot(rd, moon_dir);

    float d = 238856.0;
    float r = 1076.0;
    // Angular radius
    //float moon_size = 0.12;//20.0*atan(r/d);
    //float moon_s;// = smoothstep(TAU-moon_size, TAU-moon_size+0.01, TAU-moon_d);

    // The normal should be -moon_dir at the center,
    //	and cross(moon_dir,cross(moon_dir,rd)) at an edge
    float t2, t3;
    vec3 moon_pos = moon_dir * d * 200.0;
    bool c = isect_sphere(ro-moon_pos, rd, r*3000.0, t2, t3) && t3 >= 0.0;
    float moon_s = float(c);
    //t2 = d - r*(1.0-moon_d/moon_size);//smoothstep(0.0, moon_size, moon_d));
    vec3 p = ro + rd * t2;
    vec3 moon_n = normalize(p-moon_pos);//normalize(mix(cross(moon_dir, cross(moon_dir,rd)), -moon_dir, 0.05*cos(moon_d)));//smoothstep(TAU-moon_size,TAU,TAU-moon_d)));
    // We use the usual cosine-falloff law to generate realistic moon phases
    //	In reality, though, the moon should be offset from the sun sideways too, and the direction
    //	from the sun to moon isn't exactly the same as the direction from the sun to the earth.
    // The noise here is for craters
    float moon = max(0.6,1.0-smoothstep(0.5,1.0,1.1*fBm(3.08+3.0*(rd-moon_dir))));//*smoothstep(0.4,0.9,dot(moon_n,sun_dir));
    // These too lines are just trying to get the moon to blend better with the daytime sky.
    //	It still looks bad, more work is needed.
    vec3 moon_c = 0.8*max(mix(vec3(0.0), vec3(moon),moon_s),0.0);
    stars = mix(stars, vec3(0.0), moon_s);
    //col = col*0.2+max(col*0.8,moon_c);

    /*float moon_power = 2.0;// - sun_height;
    float phaseR_moon = rayleigh_phase_func(mu_moon);
    float phaseM_moon = henyey_greenstein_phase_func(mu_moon);
    // optical depth (or "average density")
	// represents the accumulated extinction coefficients
	// along the path, multiplied by the length of that path
	optical_depthR = 0.;
	optical_depthM = 0.;

	sumR = vec3(0);
	sumM = vec3(0);
	march_pos = 0.;

	for (int i = 0; i < num_samples; i++) {
		vec3 s =
			ro +
			rd * (march_pos + 0.5 * march_step);
		float height = length(s) - earth_radius;

		// integrate the height scale
		float hr = exp(-height / hR) * march_step;
		float hm = exp(-height / hM) * march_step;
		optical_depthR += hr;
		optical_depthM += hm;

		// gather the sunlight
		float optical_depth_lightR = 0.;
		float optical_depth_lightM = 0.;
		bool overground = get_sun_light(
			s, moon_dir,
			optical_depth_lightR,
			optical_depth_lightM);

		if (overground) {
			vec3 tau =
				betaR * (optical_depthR + optical_depth_lightR) +
				betaM * 1.1 * (optical_depthM + optical_depth_lightM);
			vec3 attenuation = exp(-tau);

			sumR += hr * attenuation;
			sumM += hm * attenuation;
		}

		march_pos += march_step;
	}

    //float moon_r = max(0.0, dot(moon_n, -sun_dir));

    // "The reciprocity principle in lunar photometry", Minnaert
    float k = 1.0;
    float moon_q = dot(moon_n, sun_dir) * dot(moon_n, -rd);
    float moon_r = mix(max(0.0,dot(moon_n,sun_dir)), max(0.0,(k+1.0)/TAU * max(0.04,sign(moon_q))*pow(moon_q, 0.3)), 0.6);

    vec3 moon_col = (moon_r *
		3.0 * moon_c + max(0.0, dot(-moon_dir, sun_dir))) * moon_power *
		(sumR * phaseR_moon * betaR +
		sumM * phaseM_moon * betaM * (1.0 + cloud_e));*/
	vec3 moon_col = moon_c * dot(moon_n, sun_dir);
    //col += abs(moon_n) * moon_s;//moon_col;
    col += max(vec3(0.0),moon_col);//mix(col, moon_col, moon_s);

    col += stars;
    //col += cloud_fac;
    col = mix(col,vec3(1.0),cloud_fac);

    return col;
}
