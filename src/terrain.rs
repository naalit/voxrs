use super::chunk::Chunks;
use super::common::*;
use super::glm::*;
use super::material::*;
use noise::*;
use rayon::prelude::*;

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
                    c.chunks[i] =
                        self.gen_chunk(ivec3(x as i32, y as i32, z as i32) - CHUNK_NUM as i32 / 2);
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
                    ) as Block;
                }
            }
        });
        c
    }

    fn gen_block(&self, loc: Vector3<f32>) -> Material {
        let h = self.height(vec2(loc.x, loc.z));
        let surface = if h < 2.0 {
            Material::Sand
        } else {
            Material::Grass
        };
        
        if abs(loc.y - h) < 1.0 {
            surface
        } else if loc.y < h {
            Material::Stone
        } else if loc.y < 1.0 {
            Material::Water
        } else {
            Material::Air
        }
    }

    fn height(&self, loc: Vector2<f32>) -> f32 {
        12.0 * self.noise.get(*(to_dvec2(loc) * 0.005).as_array()) as f32
    }
}
