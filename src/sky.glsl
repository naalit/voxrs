bool isect(vec3 ro, vec3 rd, vec3 p, float size) {
    vec3 mn = p - size;
    vec3 mx = p + size;
    vec3 t1 = (mn-ro) / rd;
    vec3 t2 = (mx-ro) / rd;
    vec3 tmin = min(t1, t2);
    vec3 tmax = max(t1, t2);
	vec2 t = vec2(max(tmin.x, max(tmin.y, tmin.z)), min(tmax.x, min(tmax.y, tmax.z)));
    return (t.y > t.x) && (t.y > 0.0);
}
#define TAU 6.28
#define SUN_SPEED 0.1
#define iTime 2.0
vec3 sky(vec3 ro, vec3 rd)
{
    float sun_speed = TAU * SUN_SPEED;
    vec3 sun_dir = vec3(sin(iTime * sun_speed), cos(iTime * sun_speed), 0.0);

    float sun = float(isect(ro, rd, ro + sun_dir * 1000.0, 50.0));
	sun += 0.3 * float(isect(ro, rd, ro + sun_dir * 1000.0, 100.0));

    vec3 col = vec3(sun) * pow(vec3(0.7031,0.4687,0.1055), vec3(1.5))
		+ 0.8 * vec3(0.3984,0.5117,0.7305) * ((0.5 + 1.0 * pow(sun_dir.y,0.4)) * (1.5-dot(vec3(0,1,0), rd)));

    return pow(col, vec3(2.2));
}
