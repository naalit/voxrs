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

// Really a BSDF
vec3 brdf(vec3 from, vec3 to, vec3 n, MatData mat) {
    float ior = mat.ior; //1.5;
    float ior_i = 1.0/ior;
    float nDotL = dot(n,to);
    bool is_refract = mat.trans > 0.0 && nDotL <= 0.0;

    // Half vector
    vec3 h = normalize(from + to);
    if (is_refract)
        h = to + ior_i * from;

    // Schlick fresnel
    float f0 = (1.-ior)/(1.+ior);
    f0 *= f0;
    float vDotH = dot(from, h);
    float lDotH = dot(to, h);
    float fresnel = f0 + (1.-f0)*pow(1.-vDotH, 5.);

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


    if (is_refract)
        return
            ior_i*ior_i* dist * geometry * (1.0 - fresnel) * abs(vDotH) * abs(lDotH) /
            (sqr(vDotH + ior_i * lDotH) * dot(n, from) * nDotL)
            ;
    else
        return saturate((fresnel*geometry*dist)/(4.*dot(n, from)*nDotL)
            + (1.-f0)*oren_nayar(from, to, n, mat));
}

float voxel(vec3 pos) {
    ivec3 cpos = ivec3(pos - chunk_origin + float(scene_size)*0.5);
    return min(1.0,float(getVoxel(cpos)));
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
    // The largest component of the vector from the center to the point on the surface,
    //	is necessarily the normal.
    vec3 p = ro+rd*t.x;
    vec3 n = (p - pos);
    n = sign(n) * (abs(n.x) > abs(n.y) ? // Not y
        (abs(n.x) > abs(n.z) ? vec3(1., 0., 0.) : vec3(0., 0., 1.)) :
    	(abs(n.y) > abs(n.z) ? vec3(0., 1., 0.) : vec3(0., 0., 1.)));

    MatData mat = mat_lookup(m);

    vec3 lightDir = major_dir();
    vec3 c = sky(p,lightDir);
    vec2 tc =
        ( fract( p.yz ) * mask.x ) +
        ( fract( p.zx ) * mask.y ) +
        ( fract( p.xy ) * mask.z );
    vec3 behind = vec3(0.0);
    if (mat.trans > 0.0) {
        vec2 t_;
        vec3 pos_;
        float size_;
        int iter_ = 4;
        vec3 n_;

        vec3 dir = refract(rd,n,1.0/mat.ior);
        uint mat_index = trace(p, dir, m, t_, pos_, iter_, size_, n_);
        MatData mat_ = mat_lookup(mat_index);

        vec3 p_ = p+dir*t_.x;
        vec3 old = n;
        n = (p_ - pos_);
        n = sign(n) * (abs(n.x) > abs(n.y) ? // Not y
            (abs(n.x) > abs(n.z) ? vec3(1., 0., 0.) : vec3(0., 0., 1.)) :
        	(abs(n.y) > abs(n.z) ? vec3(0., 1., 0.) : vec3(0., 0., 1.)));

        behind = brdf(-lightDir,-dir,n,mat_);
        behind = mix(vec3(0.0),brdf(-dir,-rd,old,mat)*behind,mat.trans);
        n = old;
    }
    vec3 ref = vec3(0.0);
    if (mat.roughness < 0.2) {
        vec2 t_;
        vec3 pos_;
        float size_;
        int iter_ = 4;
        vec3 n_;

        vec3 dir = reflect(rd,n);
        uint mat_index = trace(p, dir, m, t_, pos_, iter_, size_, n_);

        vec3 old = n;
        if (mat_index != 0.0) {
            MatData mat_ = mat_lookup(mat_index);

            vec3 p_ = p+dir*t_.x;
            vec2 tc_ =
                ( fract( p_.yz ) * n_.x ) +
                ( fract( p_.zx ) * n_.y ) +
                ( fract( p_.xy ) * n_.z );

            n = (p_ - pos_);
            n = sign(n) * (abs(n.x) > abs(n.y) ? // Not y
                (abs(n.x) > abs(n.z) ? vec3(1., 0., 0.) : vec3(0., 0., 1.)) :
            	(abs(n.y) > abs(n.z) ? vec3(0., 1., 0.) : vec3(0., 0., 1.)));

            vec3 c_ = sky(p_,lightDir);

            ref = (0.1+ao(pos_,n,tc_))*mat_.color + c_*brdf(-lightDir,-dir,n,mat_);
        } else
            ref = sky(p, dir);
        ref = IPI * ref;
        ref = brdf(dir,-rd,old,mat)*ref;
        ref *= pow(1.0-mat.roughness,2.0);
        n = old;
    }
    float shadow = shadow(p,lightDir,m);
    return
        (0.1+ao(pos,n,tc)*0.2)*mat.color + shadow*c*brdf(-lightDir, -rd, n, mat) + behind + ref;
}
