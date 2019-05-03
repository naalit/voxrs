use super::common::*;
use super::glm::*;
use noise::*;
use rayon::prelude::*;
use super::chunk::Chunks;

pub struct Gen {
    noise: HybridMulti,
}

impl Gen {
    pub fn new() -> Self {
        Gen {
            noise: HybridMulti::new(),
        }
    }

    pub fn gen_chunks(&self) -> Chunks {
        let mut c = Chunks::new();
        let mut i = 0;
        for (z, page) in c.map.iter_mut().enumerate() {
            for (y, row) in page.iter_mut().enumerate() {
                for (x, n) in row.iter_mut().enumerate() {
                    c.chunks[i] = self.gen_chunk(ivec3(x as i32, y as i32, z as i32) - CHUNK_NUM as i32 / 2);
                    *n = i;
                    i += 1;
                }
            }
        }
        c
    }

    pub fn gen_chunk(&self, loc: Vector3<i32>) -> Chunk {
        let mut c = [[[0; CHUNK_SIZE]; CHUNK_SIZE]; CHUNK_SIZE];
        let rad = (CHUNK_SIZE as f32) / 2.0;
        //if loc == ivec3(0,0,0) { println!("Why is zero?"); }
        c.par_iter_mut().enumerate().for_each(|(z, row_x)| {
            for (y, row_y) in row_x.iter_mut().enumerate() {
                for (x, b) in row_y.iter_mut().enumerate() {
                    *b = self.gen_block(
                        to_vec3(loc) * (CHUNK_SIZE as f32)
                            + 0.5
                            + vec3((z as f32) - rad, (y as f32) - rad, (x as f32) - rad),
                    );
                }
            }
        });
        c
    }

    fn gen_block(&self, loc: Vector3<f32>) -> Block {
        let h = self.height(vec2(loc.x, loc.z));
        if abs(loc.y-h) < 1.0 {
            2
        } else if loc.y < h {
            1
        } else {
            0
        }
    }

    fn height(&self, loc: Vector2<f32>) -> f32 {
        12.0 * self.noise.get(*(to_dvec2(loc) * 0.025).as_array()) as f32
    }
}
