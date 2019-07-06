extern crate noise;
use crate::common::*;
use glm::*;
use noise::*;
// use rayon::prelude::*;
use std::collections::HashMap;

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
        let grid: Vec<Vec<Vec<Material>>> = (0..CHUNK_SIZE as usize)
            .map(move |x| {
                (0..CHUNK_SIZE as usize)
                    .map(move |y| (x, y))
                    .map(move |(x, y)| {
                        (0..CHUNK_SIZE as usize)
                            .map(move |z| {
                                let height = 3.0
                                    + sin((start.x + x as f32) * 0.3) * 2.0
                                    + sin((start.z + z as f32) * 0.3) * 2.0;
                                if (y as f32 + start.y) < height {
                                    1
                                } else {
                                    AIR
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
