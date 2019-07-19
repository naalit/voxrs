use crate::common::*;

pub trait ChunkTrait: std::fmt::Debug + Clone {
    fn empty() -> Self;
    fn full(f: &Fn(UVec3) -> Material) -> Self;

    fn block(&self, idx: UVec3) -> Material;
    fn set_block(&mut self, idx: UVec3, new: Material);

    /// Returns the visible faces, indexed by faces[axis_i][u_i][v_i]
    ///     where u = (axis + 1) % 3; and v = (axis + 2) % 3;
    fn cull_faces(&self, axis: usize, neighbors: (&Self, &Self), phase2: bool) -> Vec<Vec<Vec<Material>>>;
}

#[derive(Debug, Clone)]
pub struct FlatChunk {
    /// Indexed by `blocks[y * CHUNK_SIZE * CHUNK_SIZE + x * CHUNK_SIZE + z]` for cache friendliness
    blocks: Vec<Material>,
}

const CHUNK_U: usize = CHUNK_SIZE as usize;

impl ChunkTrait for FlatChunk {
    fn empty() -> Self {
        FlatChunk {
            blocks: Vec::new(),
        }
    }
    fn full(f: &Fn(UVec3) -> Material) -> Self {
        let blocks = (0..CHUNK_U)
            .flat_map(|y| (0..CHUNK_U).map(move |x| (y,x)))
            .flat_map(|(y,x)| (0..CHUNK_U).map(move |z| f(UVec3::new(x,y,z))))
            .collect();
        FlatChunk {
            blocks,
        }
    }

    fn block(&self, idx: UVec3) -> Material {
        self.blocks[idx.y * CHUNK_U * CHUNK_U + idx.x * CHUNK_U + idx.z]
    }
    fn set_block(&mut self, idx: UVec3, new: Material) {
        self.blocks[idx.y * CHUNK_U * CHUNK_U + idx.x * CHUNK_U + idx.z] = new;
    }

    fn cull_faces(&self, axis: usize, neighbors: (&Self, &Self), phase2: bool) -> Vec<Vec<Vec<Material>>> {
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
            let b = neighbors.0;
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
                        culled[d_i][u_i].push(if (l == Material::Air || (!l.phase2() && phase2)) && b.phase2() == phase2 {
                            b
                        } else if b == Material::Air || (b.phase2() && !phase2) {
                            l
                        } else {
                            Material::Air
                        });
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
