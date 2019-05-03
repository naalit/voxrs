
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
