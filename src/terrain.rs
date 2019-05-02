use super::chunk::*;
use super::octree::*;
use glm::*;
use rayon::prelude::*;
use std::collections::HashMap;

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
        if r > Bailout {
            break;
        }

        // convert to polar coordinates
        let mut theta = acos(z.z / r);
        // #if ANIMATE
        // theta += iTime*0.5;
        // #endif
        let mut phi = atan(z.y / z.x);
        // #if ANIMATE
        // phi += iTime*0.5;
        // #endif
        dr = pow(r, Power - 1.0) * Power * dr + 1.0;

        // scale and rotate the point
        let zr = pow(r, Power);
        theta = theta * Power;
        phi = phi * Power;

        // convert back to cartesian coordinates
        z = vec3(sin(theta) * cos(phi), sin(phi) * sin(theta), cos(theta)) * zr;
        z = z + pos;
    }
    return 0.5 * log(r) * r / dr;
}

struct ST {
    parent: usize,
    idx: Vector3<f32>,
    pos: Vector3<f32>,
    scale: i32,
}
/*
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
*/

pub fn generate() -> Octree {
    let mut chunks: HashMap<(i32, i32, i32), Octree> = HashMap::new();
    for x in -CHUNK_NUM..CHUNK_NUM {
        for y in -CHUNK_NUM..CHUNK_NUM {
            for z in -CHUNK_NUM..CHUNK_NUM {
                let chunk = gen_chunk(ivec3(x, y, z));
                chunks.insert((x, y, z), chunk_to_tree(chunk));
            }
        }
    }
    /*

    -2  -1  0   1
-2  /   /   /   /
-1  /   /   /   /
0   /   /   /   /
1   /   /   /   /

    -1  0
-1  /   /
 0  /   /

    */
    for i in 0.. {
        if chunks.len() < 16 {
            break;
        };
        let mut chunks_2 = HashMap::new();
        let n = (chunks.len() as f32).cbrt() / 4.0;
        let n = n as i32;
        for x in -n..n {
            for y in -n..n {
                for z in -n..n {
                    chunks_2.insert(
                        (x, y, z),
                        combine_trees([
                            chunks.get(&(x * 2, y * 2, z * 2)).unwrap().clone(), // 0b000
                            chunks.get(&(x * 2, y * 2, z * 2 + 1)).unwrap().clone(), // 0b001
                            chunks.get(&(x * 2, y * 2 + 1, z * 2)).unwrap().clone(), // 0b010
                            chunks.get(&(x * 2, y * 2 + 1, z * 2 + 1)).unwrap().clone(), // 0b011
                            chunks.get(&(x * 2 + 1, y * 2, z * 2)).unwrap().clone(), // 0b100
                            chunks.get(&(x * 2 + 1, y * 2, z * 2 + 1)).unwrap().clone(), // 0b101
                            chunks.get(&(x * 2 + 1, y * 2 + 1, z * 2)).unwrap().clone(), // 0b110
                            chunks
                                .get(&(x * 2 + 1, y * 2 + 1, z * 2 + 1)) // 0b111
                                .unwrap()
                                .clone(),
                        ]),
                    );
                }
            }
        }
        chunks = chunks_2;
    }
    combine_trees([
        chunks.get(&(-1, -1, -1)).unwrap().clone(),
        chunks.get(&(-1, -1, 0)).unwrap().clone(),
        chunks.get(&(-1, 0, -1)).unwrap().clone(),
        chunks.get(&(-1, 0, 0)).unwrap().clone(),
        chunks.get(&(0, -1, -1)).unwrap().clone(),
        chunks.get(&(0, -1, 0)).unwrap().clone(),
        chunks.get(&(0, 0, -1)).unwrap().clone(),
        chunks.get(&(0, 0, 0)).unwrap().clone(),
    ])
}

pub fn gen_chunk(loc: Vector3<i32>) -> [[[usize; CHUNK_SIZE]; CHUNK_SIZE]; CHUNK_SIZE] {
    let mut c = [[[0; CHUNK_SIZE]; CHUNK_SIZE]; CHUNK_SIZE];
    let rad = (CHUNK_SIZE as f32) / 2.0;
    //if loc == ivec3(0,0,0) { println!("Why is zero?"); }
    for (x, row_x) in c.iter_mut().enumerate() {
        for (y, row_y) in row_x.iter_mut().enumerate() {
            for (z, b) in row_y.iter_mut().enumerate() {
                *b = gen_block(
                    to_vec3(loc) * (CHUNK_SIZE as f32)
                        + 0.5
                        + vec3((x as f32) - rad, (y as f32) - rad, (z as f32) - rad),
                );
            }
        }
    }
    c
}

pub fn gen_block(loc: Vector3<f32>) -> usize {
    let h = height(vec2(loc.x, loc.z));
    if loc.y <= h {
        1
    } else {
        0
    }
    // ;
    // if loc.y == 0.5 && loc.x == 4.5 {
    //     1
    // } else {
    //     0
    // }
}

pub fn height(loc: Vector2<f32>) -> f32 {
    sin(loc.x) * 4.0//cos(loc.y) * 4.0
}
