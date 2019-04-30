use glm::*;
use super::octree::*;

// These are parameters for the Mandelbulb, change if you want. Higher is usually slower
const Power: f32 = 4.0;
const Bailout: f32 = 1.5;
const Iterations: i32 = 6;

// This is from http://blog.hvidtfeldts.net/index.php/2011/09/distance-estimated-3d-fractals-v-the-mandelbulb-different-de-approximations/
// 	because it had code that I could just copy and paste
fn dist(pos: Vector3<f32>) -> f32 {
    // This function takes ~all of the rendering time, the trigonometry is super expensive
    // So if there are any faster approximations, they should definitely be used
	let mut z = pos;
	let mut dr = 1.0;
	let mut r = 0.0;
    for i in 0..Iterations {
		r = length(z);
		if r>Bailout { break; }

		// convert to polar coordinates
		let mut theta = acos(z.z/r);
        // #if ANIMATE
        // theta += iTime*0.5;
        // #endif
		let mut phi = atan(z.y/z.x);
        // #if ANIMATE
        // phi += iTime*0.5;
        // #endif
		dr = pow( r, Power-1.0)*Power*dr + 1.0;

		// scale and rotate the point
		let zr = pow( r,Power);
		theta = theta*Power;
		phi = phi*Power;

		// convert back to cartesian coordinates
		z = vec3(sin(theta)*cos(phi), sin(phi)*sin(theta), cos(theta)) * zr;
        z = z + pos;
	}
	return 0.5*log(r)*r/dr;
}

struct ST {
    parent: usize,
    idx: Vector3<f32>,
    pos: Vector3<f32>,
    scale: i32,
}

pub fn generate() -> Octree {
    let levels = 6;
    let mut stack: Vec<ST> = vec![];

    let mut tree: Octree = Vec::new();
    for i in 0.. {
        let (pos, root, idx, parent, scale) =
            if i == 0 { (vec3(0.0,0.0,0.0), true, vec3(0.0,0.0,0.0), 0, 0) }
            else if !stack.is_empty() { let s = stack.pop().unwrap(); (s.pos, false, s.idx, s.parent, s.scale) }
            else { break };

            let mut v = Node { leaf: [true; 8], pointer: [0; 8] };
            let size = 2.0_f32.powf(-scale as f32) * 4.0;
            for j in 0..8 {
                let jdx = Node::position(j);
                let np = pos + jdx * size * 0.5;

                let d = dist(np);
                if scale >= levels {
                    v.leaf[j] = true;
                    if d > size * 0.5 {
                        v.pointer[j] = 0;
                    } else {
                        v.pointer[j] = 1;
                    }
                } else if d > size {
                    v.leaf[j] = true;
                    v.pointer[j] = 0;
                } else if d < -size {
                    v.leaf[j] = true;
                    v.pointer[j] = 1;
                } else {
                    stack.push(ST{parent: i, idx: jdx, pos: np, scale: scale+1 });
                }
            }
            if !root {
                let uidx = Node::idx(idx);
                tree[parent].leaf[uidx] = false;
                tree[parent].pointer[uidx] = i;
            }
            tree.push(v);
    };
    tree
}
