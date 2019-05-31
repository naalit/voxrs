#define BEVEL 0

vec3 shade(in vec3 ro, in vec3 rd, in vec2 t, in vec3 pos) {
    vec3 p = ro+rd*t.x;
    vec3 n = p-pos;

    //n = sign(n) * vec3(all(greaterThan(n.xx, n.yz)), all(greaterThan(n.yy,n.xz)), all(greaterThan(n.zz,n.yx)));

#if BEVEL
    n = normalize(sign(n) * pow(abs(n), vec3(3)));
#else
    n = sign(n) * (abs(n.x) > abs(n.y) ? // Not y
        (abs(n.x) > abs(n.z) ? vec3(1., 0., 0.) : vec3(0., 0., 1.)) :
    	(abs(n.y) > abs(n.z) ? vec3(0., 1., 0.) : vec3(0., 0., 1.)));
#endif

    vec3 light = reflect(rd,n);//-normalize(vec3(-0.2,0.8,0.3));
    return vec3(1.0) * dot(light,n);
}
