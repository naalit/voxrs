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
        let start = pos.map(|x| x as f32 * CHUNK_SIZE);

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
        if start.y > 0.0 && start.y > chunk_heightmap.iter().flatten().max_by_key(|x| x.ceil() as i32).unwrap().ceil() {
            return Chunk::empty();
        }

        let grid = Chunk::full(&mut |p| {
            let height = chunk_heightmap[p.x][p.z]; //3.0 + 4.0 * self.noise.get([(start.x as f64 + x as f64) * 0.01, (start.z as f64 + z as f64) * 0.01]) as f32;
            let y = p.y as f32 + start.y;
            if y == height.ceil() {
                Material::Grass
            } else if y < height
                && y > height - 3.0
            {
                Material::Dirt
            } else if y < height {
                Material::Stone
            } else if y < 0.0 {
                Material::Water
            } else {
                Material::Air
            }
        });

        grid
    }
}
