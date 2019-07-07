extern crate noise;
use crate::common::*;
use glm::*;
use noise::*;
// use rayon::prelude::*;

pub struct Gen {
    noise: HybridMulti,
}

impl Gen {
    pub fn new() -> Self {
        Gen {
            noise: HybridMulti::new().set_seed(1),
        }
    }

    pub fn gen(&self, pos: IVec3) -> Chunk {
        let start = to_vec3(pos) * CHUNK_SIZE;

        let chunk_heightmap = (0..CHUNK_SIZE as usize).map(move |x| {
            (0..CHUNK_SIZE as usize)
                .map(move |z| {
                    3.0 + 12.0 * self.noise.get([(start.x as f64 + x as f64) * 0.01, (start.z as f64 + z as f64) * 0.01]) as f32
                }).collect::<Vec<_>>()
        }).collect::<Vec<_>>();

        let grid: Vec<Vec<Vec<Material>>> = (0..CHUNK_SIZE as usize)
            .map(move |x| {
                (0..CHUNK_SIZE as usize)
                    .map(move |y| (x, y))
                    .map(|(x, y)| {
                        (0..CHUNK_SIZE as usize)
                            .map(|z| {
                                let height = chunk_heightmap[x][z]; //3.0 + 4.0 * self.noise.get([(start.x as f64 + x as f64) * 0.01, (start.z as f64 + z as f64) * 0.01]) as f32;
                                if (y as f32 + start.y) == ceil(height) {
                                    Material::Grass
                                } else if (y as f32 + start.y) < height && (y as f32 + start.y) > height-3.0 {
                                    Material::Dirt
                                } else if (y as f32 + start.y) < height {
                                    Material::Stone
                                } else {
                                    Material::Air
                                }
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();
        grid
    }
}
