extern crate noise;
use noise::*;
//use super::chunk::*;
use glm::*;
// use rayon::prelude::*;
use std::collections::HashMap;

#[derive(Copy, Clone)]
pub struct Node { pointer: [u32; 8] }
implement_uniform_block!(Node, pointer);

impl Node {
    /// Converts between a 3D vector representing the child slot, and the actual index into the `leaf` and `pointer` arrays
    pub fn idx<T: BaseNum>(idx: Vector3<T>) -> usize {
        // Once again, this function closely mirrors the GLSL one for testing
        let mut ret = 0;
        ret |= usize::from(idx.x > T::zero()) << 2;
        ret |= usize::from(idx.y > T::zero()) << 1;
        ret |= usize::from(idx.z > T::zero());
        ret
    }

    /// Converts between a 3D vector representing the child slot, and the actual index into the `leaf` and `pointer` arrays
    pub fn position(idx: usize) -> Vector3<f32> {
        vec3(
            if idx & (1 << 2) > 0 { 1.0 } else { -1.0 },
            if idx & (1 << 1) > 0 { 1.0 } else { -1.0 },
            if idx & 1 > 0 { 1.0 } else { -1.0 },
        )
    }
}

pub struct Gen {
    noise: HybridMulti,
}

impl Gen {
    pub fn new() -> Self {
        Gen {
            noise: HybridMulti::new(),
        }
    }
    pub fn dist(&self, x: Vec3) -> f32 {
        x.y + 1.0 * self.noise.get([x.x as f64 * 0.1, x.z as f64 * 0.1]) as f32
    }
}

struct ST {
    parent: usize,
    idx: Vector3<f32>,
    pos: Vector3<f32>,
    scale: i32,
}

pub fn generate() -> Vec<Node> {
    let gen = Gen::new();

    let levels = 6;
    let mut stack: Vec<ST> = vec![];

    let mut tree: Vec<Node> = Vec::new();
    for i in 0.. {
        let (pos, root, idx, parent, scale) =
            if i == 0 { (vec3(0.0,0.0,0.0), true, vec3(0.0,0.0,0.0), 0, 0) }
            else if !stack.is_empty() { let s = stack.pop().unwrap(); (s.pos, false, s.idx, s.parent, s.scale) }
            else { break };

            let mut v = Node { pointer: [0; 8] };
            let size = 2.0_f32.powf(-scale as f32) * 2.0;
            for j in 0..8 {
                let jdx = Node::position(j);
                let np = pos + jdx * size * 0.5;

                let d = gen.dist(np);
                if scale >= levels {
                    if d > size * 0.5 {
                        v.pointer[j] = 0;
                    } else {
                        v.pointer[j] = 0b10;
                    }
                } else if d > size {
                    //v.leaf[j] = true;
                    v.pointer[j] = 0;
                } else if d < -size {
                    //v.leaf[j] = true;
                    v.pointer[j] = 0b10;
                } else {
                    stack.push(ST{parent: i, idx: jdx, pos: np, scale: scale+1 });
                }
            }
            if !root {
                let uidx = Node::idx(idx);
                tree[parent].pointer[uidx] = ((i as u32) << 1) | 1;
            }
            tree.push(v);
    };
    tree
}

/*
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
*/
