extern crate noise;
use crate::common::*;
use noise::*;
// use rayon::prelude::*;

pub struct Gen {
    noise: HybridMulti,
}

impl Gen {
    pub fn new() -> Self {
        Gen {
            noise: HybridMulti::new().set_seed(1).set_octaves(8).set_persistence(0.5),
        }
    }

    pub fn gen(&self, pos: IVec3) -> Chunk {
        let start = pos.map(|x| x * CHUNK_SIZE as i32);

        let chunk_heightmap = (0..CHUNK_SIZE as usize)
            .map(move |x| {
                (0..CHUNK_SIZE as usize)
                    .map(move |z| {
                        3.0 + 48.0
                            * self.noise.get([
                                (start.x as f64 + x as f64) * 0.0004,
                                (start.z as f64 + z as f64) * 0.0004,
                            ]) as f32
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        // The whole chunk is above the ground, so we don't need to bother
        if start.y > 0 && start.y > chunk_heightmap.iter().flatten().map(|x| x.ceil() as i32).max().unwrap() {
            return Chunk::empty();
        }

        Chunk::full(&|p| {
            let height = chunk_heightmap[p.x][p.z]; //3.0 + 4.0 * self.noise.get([(start.x as f64 + x as f64) * 0.01, (start.z as f64 + z as f64) * 0.01]) as f32;
            let y = p.y as i32 + start.y;
            if y == height.ceil() as i32 {
                Material::Grass
            } else if y < height.ceil() as i32
                && y > height as i32 - 3
            {
                Material::Dirt
            } else if y < height as i32 {
                Material::Stone
            } else if y < 0 {
                Material::Water
            } else {
                Material::Air
            }
        })
    }
}
