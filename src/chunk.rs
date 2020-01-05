use crate::common::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Chunk {
    /// Indexed by `blocks[y * CHUNK_SIZE * CHUNK_SIZE + x * CHUNK_SIZE + z]` for cache friendliness
    Flat(Vec<Material>),
    /// Same index. (length, mat)
    Runs(Vec<(u16, Material)>),
}

const CHUNK_U: usize = CHUNK_SIZE as usize;

impl Chunk {
    pub fn empty() -> Self {
        Chunk::Runs(vec![((CHUNK_U * CHUNK_U * CHUNK_U) as u16, Material::Air)])
    }
    pub fn full(f: &Fn(UVec3) -> Material) -> Self {
        let mut runs: Vec<(u16, Material)> = Vec::new();
        let mut blocks: Vec<Material> = Vec::new();
        for y in 0..CHUNK_U {
            for x in 0..CHUNK_U {
                for z in 0..CHUNK_U {
                    let b = f(UVec3::new(x, y, z));
                    if runs.last().map_or(false, |x| x.1 == b) {
                        runs.last_mut().unwrap().0 += 1;
                    } else {
                        runs.push((1, b));
                    }
                    blocks.push(b);
                }
            }
        }
        if runs.len() <= 16 {
            Chunk::Runs(runs)
        } else {
            Chunk::Flat(blocks)
        }
    }

    pub fn block(&self, idx: UVec3) -> Material {
        match self {
            Chunk::Flat(ref blocks) => blocks[idx.y * CHUNK_U * CHUNK_U + idx.x * CHUNK_U + idx.z],
            Chunk::Runs(ref runs) => {
                let mut ret = Material::Air;
                let i = idx.y * CHUNK_U * CHUNK_U + idx.x * CHUNK_U + idx.z;
                let mut c = 0;
                for (l, b) in runs {
                    c += l;
                    if i < c.into() {
                        ret = *b;
                        break;
                    }
                }
                ret
            }
        }
    }
    pub fn set_block(&mut self, idx: UVec3, new: Material) {
        match self {
            Chunk::Flat(ref mut blocks) => {
                blocks[idx.y * CHUNK_U * CHUNK_U + idx.x * CHUNK_U + idx.z] = new;
            }
            Chunk::Runs(ref mut runs) => {
                let i = idx.y * CHUNK_U * CHUNK_U + idx.x * CHUNK_U + idx.z;
                let mut c = 0;
                for j in 0..runs.len() {
                    let (l, b) = runs[j];
                    c += l;
                    if i < c.into() {
                        let start = c - l;
                        let new_len = i as u16 - start;
                        runs[j].0 = new_len;
                        runs.insert(j + 1, (1, new));
                        let after = c - (i + 1) as u16;
                        if after != 0 {
                            runs.insert(j + 2, (after, b));
                        }
                        break;
                    }
                }
                if runs.len() > 16 {
                    // RLE is no longer suitable for this chunk, switch to flat encoding
                    self.flatten();
                }
            }
        }
    }

    fn flatten(&mut self) {
        if let Chunk::Runs(runs) = self {
            let mut blocks = Vec::new();
            // We can do this because the flat array is in the same order as the runs
            for (len, b) in runs {
                blocks.extend((0..*len).map(|_| b.clone()));
            }
            *self = Chunk::Flat(blocks);
        }
    }

    /// Indexed by `faces[axis][u][v]` where `u = (axis + 1) % 3; v = (axis + 2) % 3;`
    pub fn cull_faces(
        &self,
        axis: usize,
        neighbors: (&Self, &Self),
        phase2: bool,
    ) -> Vec<Vec<Vec<Material>>> {
        // Special case
        if let Chunk::Runs(runs) = self {
            if runs.len() == 1 {
                let r = runs.first().unwrap();
                if r.1 == Material::Air {
                    return Vec::new();
                }
            }
        }

        let u = (axis + 1) % 3;
        let v = (axis + 2) % 3;

        let mut last: Vec<Vec<Material>> = {
            let f = neighbors.0;
            (0..CHUNK_U)
                .map(|u_i| {
                    (0..CHUNK_U)
                        .map(|v_i| {
                            let mut idx = UVec3::zeros();
                            idx[axis] = CHUNK_U - 1;
                            idx[u] = u_i;
                            idx[v] = v_i;
                            let b = f.block(idx);
                            if !b.phase2() || phase2 {
                                b
                            } else {
                                Material::Air
                            }
                        })
                        .collect()
                })
                .collect()
        };

        let end: Vec<Vec<Material>> = {
            let b = neighbors.1;
            (0..CHUNK_U)
                .map(|u_i| {
                    (0..CHUNK_U)
                        .map(|v_i| {
                            let mut idx = UVec3::zeros();
                            idx[axis] = 0;
                            idx[u] = u_i;
                            idx[v] = v_i;
                            b.block(idx)
                        })
                        .collect()
                })
                .collect()
        };

        let mut culled = Vec::new();
        for d_i in 0..CHUNK_U + 1 {
            culled.push(Vec::new());
            for u_i in 0..CHUNK_U {
                culled[d_i].push(Vec::new());
                for v_i in 0..CHUNK_U {
                    if d_i < CHUNK_U {
                        let mut idx = UVec3::zeros();
                        idx[axis] = d_i;
                        idx[u] = u_i;
                        idx[v] = v_i;
                        let b = self.block(idx);
                        let l = last[u_i][v_i];
                        culled[d_i][u_i].push(
                            if (l == Material::Air || (!l.phase2() && phase2))
                                && b.phase2() == phase2
                            {
                                b
                            } else if b == Material::Air || (b.phase2() && !phase2) {
                                l
                            } else {
                                Material::Air
                            },
                        );
                        last[u_i as usize][v_i as usize] = if b.phase2() == phase2 {
                            b
                        } else {
                            Material::Air
                        };
                    } else {
                        // The last edge
                        let l = last[u_i][v_i];
                        let b = end[u_i][v_i];
                        culled[d_i][u_i].push(if b == Material::Air || (b.phase2() && !phase2) {
                            l
                        } else {
                            Material::Air
                        });
                    }
                }
            }
        }

        culled
    }
}
