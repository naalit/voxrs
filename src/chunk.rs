use super::common::*;
use super::glm::*;
use super::terrain::*;
// use glium::backend::Facade;
use std::sync::mpsc::*;
// use std::sync::Arc;

/// Holds the world, and manages GPU storage and the like
pub struct ChunkManager {
    // chunks: Vec<Vec<Vec<(u8, u8, u8)>>>,
    /// The exact world-space origin point of the chunks
    // origin: IVec3,
    /// The world generator
    gen: Gen,
}

impl ChunkManager {
    pub fn chunk_thread(self, to: Sender<Message>, from: Receiver<Message>) {
        loop {
            let p = from.recv();
            if let Ok(p) = p {
                match p {
                    Message::LoadChunks(locs) => {
                        let mut chunks = Vec::new();
                        for l in locs {
                            chunks.push((l,self.gen.gen_chunk(l)));
                        }
                        to.send(Message::Chunks(chunks)).unwrap();
                    }
                    Message::UnloadChunks(_) => {
                        // TODO
                    }
                    _ => ()
                }
            } else { break; }
        }
    }

    // pub fn gen_host<F: Facade + ?Sized>(&self, f: &F) -> ChunkHost {
    //     ChunkHost::new(f, &self.chunks)
    // }

    /// Create a new ChunkManager. `chunks` are the starting chunks, already generated or loaded
    pub fn new() -> Self {
        let gen = Gen::new();
        // let chunks = gen.gen_chunks();
        // let chunks = chunks.to_uniform().chunks;
        ChunkManager {
            // chunks,
            // origin,
            gen,
        }
    }

    // /// Convert world-space coordinates to chunk-space
    // pub fn world_to_chunk(&self, world: IVec3) -> UVec3 {
    //     to_uvec3(world - self.origin + (CHUNK_NUM * CHUNK_SIZE / 2) as i32)
    // }
    //
    // /// Convert chunk-space coordinates to world-space
    // pub fn chunk_to_world(&self, chunk: UVec3) -> IVec3 {
    //     to_ivec3(chunk) + self.origin - (CHUNK_NUM * CHUNK_SIZE / 2) as i32
    // }

    // / Add one block
    // / Note: Coordinates are in chunk space!
    // pub fn add(&mut self, loc: UVec3, block: Block) {
    //     let chunk = loc / CHUNK_SIZE as u32;
    //     let in_chunk = loc % CHUNK_SIZE as u32;
    //     let offset = self.chunks.chunks[chunk.z as usize][chunk.y as usize][chunk.x as usize];
    //     let x = offset.0 as usize + in_chunk.x as usize;
    //     let y = offset.1 as usize + in_chunk.y as usize;
    //     let z = offset.2 as usize + in_chunk.z as usize;
    //     self.chunks.blocks[z][y][x] = block;
    // }

    // / Loads in the next row, page, or column, positive or negative.
    // / Only one component of `dir` should have a value, which should be -1 or 1
    // pub fn load(&mut self, dir: IVec3) -> CommandList {
    //     let mut chunks_load = Vec::new();
    //     let mut new_chunks = self.chunks.clone();
    //     // Advance the origin
    //     // print!("Old origin: {:?}", self.origin);
    //     self.origin = self.origin + dir * CHUNK_SIZE as i32;
    //     // World-space chunk coordinates, in chunks instead of blocks
    //     let start = self.origin / CHUNK_SIZE as i32 - CHUNK_NUM as i32 / 2;
    //     // println!("New origin: {:?}", self.origin);
    //     for (z, page) in new_chunks.iter_mut().enumerate() {
    //         for (y, row) in page.iter_mut().enumerate() {
    //             for (x, c) in row.iter_mut().enumerate() {
    //                 // Where would this chunk be in the _old_ chunks?
    //                 let p = ivec3(x as i32, y as i32, z as i32) + dir;
    //                 let n = CHUNK_NUM as i32;
    //
    //                 if p.x >= 0 && p.y >= 0 && p.z >= 0 && p.x < n && p.y < n && p.z < n {
    //                     // It's in bounds, we already have the blocks for it; just move the offset
    //                     *c = self.chunks[p.z as usize][p.y as usize][p.x as usize];
    //                     // assert!(self.origin.z != 112 || p.z != 0);
    //                 } else {
    //                     // if self.origin.z == 112 {
    //                     //     print!("Old: {:?}, new: ", p);
    //                     // }
    //                     // It's out of bounds, we need to make a new chunk and delete the old one
    //
    //                     // Wrap the coordinates around. If it's `-1`, this will be `CHUNK_NUM-1`;
    //                     //  if it's `CHUNK_NUM`, this will be `CHUNK_NUM % CHUNK_NUM = 0`.
    //                     //  And if it's something else, it won't change
    //                     let p = (p + n) % n;
    //                     // if self.origin.z == 112 {
    //                     //     println!("{:?}", p);
    //                     // }
    //                     // A now-unoccupied chunk
    //                     let i = self.chunks[p.z as usize][p.y as usize][p.x as usize];
    //                     *c = i;
    //
    //                     // Generate a new chunk and add it to `blocks`
    //                     let i = (i.2 as usize, i.1 as usize, i.0 as usize);
    //                     // let y = if self.origin.z == 112 {
    //                     //     0
    //                     // } else { y };
    //                     let new_chunk = self.gen.gen_chunk(
    //                         // World-space chunk coordinates, in chunks instead of blocks
    //                         ivec3(start.z,start.y,start.x) + ivec3(z as i32, y as i32, x as i32),
    //                     );
    //                     chunks_load.push((Arc::from(new_chunk),(i.0 as u32, i.1 as u32, i.2 as u32)));
    //                     // for (z, page) in new_chunk.iter().enumerate() {
    //                     //     for (y, row) in page.iter().enumerate() {
    //                     //         for (x, b) in row.iter().enumerate() {
    //                     //             self.chunks.blocks[i.2 + z][i.1 + y][i.0 + x] = *b;
    //                     //         }
    //                     //     }
    //                     // }
    //                 }
    //             }
    //         }
    //     }
    //     let z = Arc::from(new_chunks.clone());
    //     self.chunks = new_chunks;
    //     Some((
    //         chunks_load,
    //         z,
    //     ))
    // }
    //
    // /// Loads in new chunks if necessary, given the player position
    // pub fn update(&mut self, player: Vec3) -> CommandList {
    //     let diff = player - to_vec3(self.origin);
    //     let t = CHUNK_SIZE as f32 * 0.5;
    //     // Has the player gone more than half a chunk away from the origin (ie, left the chunk)?
    //     if abs(diff.x) > t || abs(diff.y) > t || abs(diff.z) > t {
    //         let dir = if abs(diff.x) > abs(diff.y) {
    //             if abs(diff.x) > abs(diff.z) {
    //                 ivec3(sign(diff.x) as i32, 0, 0)
    //             } else {
    //                 ivec3(0, 0, sign(diff.z) as i32)
    //             }
    //         } else {
    //             if abs(diff.y) > abs(diff.z) {
    //                 ivec3(0, sign(diff.y) as i32, 0)
    //             } else {
    //                 ivec3(0, 0, sign(diff.z) as i32)
    //             }
    //         };
    //         // println!("Loading new chunk in direction {:?}", dir);
    //         self.load(dir)
    //     } else { None }
    // }
}

pub struct Chunks {
    pub chunks: Vec<Chunk>,
    pub map: Vec<Vec<Vec<usize>>>,
}
pub struct ChunksU {
    pub chunks: Vec<Vec<Vec<(u8, u8, u8)>>>,
    pub blocks: Vec<Vec<Vec<Block>>>,
}

impl ChunksU {
    fn new() -> Self {
        ChunksU {
            chunks: Vec::new(),
            blocks: Vec::new(),
        }
    }
}

impl Chunks {
    pub fn new() -> Self {
        Chunks {
            chunks: vec![
                [[[0; CHUNK_SIZE]; CHUNK_SIZE]; CHUNK_SIZE];
                CHUNK_NUM * CHUNK_NUM * CHUNK_NUM
            ],
            map: vec![vec![vec![0; CHUNK_NUM]; CHUNK_NUM]; CHUNK_NUM],
        }
    }

    fn idx_to_pos(n: usize) -> (usize, usize, usize) {
        let q = n / CHUNK_NUM;
        (
            CHUNK_SIZE * (q / CHUNK_NUM),
            CHUNK_SIZE * (q % CHUNK_NUM),
            CHUNK_SIZE * (n % CHUNK_NUM),
        )
    }

    pub fn to_uniform(&self) -> ChunksU {
        let mut c = ChunksU::new();
        // let s = (self.chunks.len() as f32).cbrt() as usize;
        // assert_eq!(s, CHUNK_NUM);
        // assert_eq!(self.chunks.len(), CHUNK_NUM*CHUNK_NUM*CHUNK_NUM);
        c.blocks = vec![
            vec![vec![0; CHUNK_SIZE * CHUNK_NUM]; CHUNK_SIZE * CHUNK_NUM];
            CHUNK_SIZE * CHUNK_NUM
        ];
        for (n, i) in self.chunks.iter().enumerate() {
            let p = Self::idx_to_pos(n);
            for (z, row_x) in i.iter().enumerate() {
                for (y, row_y) in row_x.iter().enumerate() {
                    for (x, b) in row_y.iter().enumerate() {
                        // assert!(p.0 <= s*CHUNK_SIZE - CHUNK_SIZE, "{}", p.0);
                        c.blocks[p.2 + z][p.1 + y][p.0 + x] = *b;
                    }
                }
            }
        }
        c.chunks = vec![vec![vec![(0, 0, 0); CHUNK_NUM]; CHUNK_NUM]; CHUNK_NUM];
        for (z, row_x) in self.map.iter().enumerate() {
            for (y, row_y) in row_x.iter().enumerate() {
                for (x, &n) in row_y.iter().enumerate() {
                    let p = Self::idx_to_pos(n);
                    let p = (p.0 as u8, p.1 as u8, p.2 as u8);
                    c.chunks[z][y][x] = p;
                }
            }
        }
        c
    }
}

#[cfg(test)]
mod tests {
    use rand::prelude::*;
    use super::*;

    #[test]
    fn idx_to_pos() {
        use std::collections::HashSet;
        let mut h = HashSet::new();
        let mut r = thread_rng();
        for _ in 0..100 {
            let i = r.gen();
            let i = Chunks::idx_to_pos(i);
            assert!(!h.contains(&i));
            h.insert(i);
        }
    }

    #[test]
    fn world_chunk_recip() {
        let origin = ivec3(324, 32534, -154354);
        /// Convert world-space coordinates to chunk-space
        fn world_to_chunk(origin: IVec3, world: IVec3) -> UVec3 {
            to_uvec3(world - origin + (CHUNK_NUM * CHUNK_SIZE / 2) as i32)
        }

        /// Convert chunk-space coordinates to world-space
        fn chunk_to_world(origin: IVec3, chunk: UVec3) -> IVec3 {
            to_ivec3(chunk) + origin - (CHUNK_NUM * CHUNK_SIZE / 2) as i32
        }
        let mut r = thread_rng();
        for _ in 0..100 {
            let x = uvec3(r.gen::<u16>() as u32, r.gen::<u16>() as u32, r.gen::<u16>() as u32);
            assert_eq!(world_to_chunk(origin, chunk_to_world(origin, x)), x);
        }
    }
}
