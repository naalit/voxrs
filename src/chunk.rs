use super::common::*;
use super::glm::*;
use super::terrain::*;
use glium::backend::Facade;

/// Holds the world, and manages GPU storage and the like
pub struct ChunkManager {
    chunks: ChunksU,
    /// These two are the buffers for storing a chunk and the chunk map, respectively, to update them
    pix_buf: glium::texture::pixel_buffer::PixelBuffer<Block>,
    dpix_buf: glium::texture::pixel_buffer::PixelBuffer<(u8, u8, u8)>,
    /// The actual textures
    chunk_buf: glium::texture::unsigned_texture3d::UnsignedTexture3d,
    block_buf: glium::texture::unsigned_texture3d::UnsignedTexture3d,
    /// The exact world-space origin point of the chunks
    origin: IVec3,
    /// The world generator
    gen: Gen,
}

impl ChunkManager {
    /// Create a new ChunkManager. `chunks` are the starting chunks, already generated or loaded
    pub fn new<F: Facade + ?Sized>(f: &F, origin: IVec3) -> Self {
        let gen = Gen::new();
        let chunks = gen.gen_chunks();
        let chunks = chunks.to_uniform();
        let block_buf = glium::texture::unsigned_texture3d::UnsignedTexture3d::with_format(
            f,
            chunks.blocks.clone(),
            glium::texture::UncompressedUintFormat::U16,
            glium::texture::MipmapsOption::NoMipmap,
        )
        .unwrap();
        let chunk_buf = glium::texture::unsigned_texture3d::UnsignedTexture3d::with_format(
            f,
            chunks.chunks.clone(),
            glium::texture::UncompressedUintFormat::U8U8U8,
            glium::texture::MipmapsOption::NoMipmap,
        )
        .unwrap();
        ChunkManager {
            chunks,
            pix_buf: glium::texture::pixel_buffer::PixelBuffer::new_empty(
                f,
                CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE,
            ),
            dpix_buf: glium::texture::pixel_buffer::PixelBuffer::new_empty(
                f,
                CHUNK_NUM * CHUNK_NUM * CHUNK_NUM,
            ),
            chunk_buf,
            block_buf,
            origin,
            gen,
        }
    }

    pub fn chunks_u(&self) -> &glium::texture::unsigned_texture3d::UnsignedTexture3d {
        &self.chunk_buf
    }
    pub fn blocks_u(&self) -> &glium::texture::unsigned_texture3d::UnsignedTexture3d {
        &self.block_buf
    }

    /// Get the origin as an array, for passing it to the shader (_u_niform)
    pub fn origin_u(&self) -> [f32; 3] {
        *to_vec3(self.origin).as_array()
    }

    /// Convert world-space coordinates to chunk-space
    pub fn world_to_chunk(&self, world: IVec3) -> UVec3 {
        to_uvec3(world - self.origin + (CHUNK_NUM * CHUNK_SIZE / 2) as i32)
    }

    /// Convert chunk-space coordinates to world-space
    pub fn chunk_to_world(&self, chunk: UVec3) -> IVec3 {
        to_ivec3(chunk) + self.origin - (CHUNK_NUM * CHUNK_SIZE / 2) as i32
    }

    /// Add one block
    /// Note: Coordinates are in chunk space!
    pub fn add(&mut self, loc: UVec3, block: Block) {
        let chunk = loc / CHUNK_SIZE as u32;
        let in_chunk = loc % CHUNK_SIZE as u32;
        let offset = self.chunks.chunks[chunk.z as usize][chunk.y as usize][chunk.x as usize];
        let x = offset.0 as usize + in_chunk.x as usize;
        let y = offset.1 as usize + in_chunk.y as usize;
        let z = offset.2 as usize + in_chunk.z as usize;
        self.chunks.blocks[z][y][x] = block;
    }

    /// Loads in the next row, page, or column, positive or negative.
    /// Only one component of `dir` should have a value, which should be -1 or 1
    pub fn load(&mut self, dir: IVec3) {
        // BUG:
        // Somewhere in this function, we're overwriting a part of self.chunks.chunks that doesn't need to be overwritten,
        //  with a new chunk generated somewhere else (?)

        let mut new_chunks = self.chunks.chunks.clone();
        // Advance the origin
        // print!("Old origin: {:?}", self.origin);
        self.origin = self.origin + dir * CHUNK_SIZE as i32;
        // World-space chunk coordinates, in chunks instead of blocks
        let start = self.origin / CHUNK_SIZE as i32 - CHUNK_NUM as i32 / 2;
        // println!("New origin: {:?}", self.origin);
        for (z, page) in new_chunks.iter_mut().enumerate() {
            for (y, row) in page.iter_mut().enumerate() {
                for (x, c) in row.iter_mut().enumerate() {
                    // Where would this chunk be in the _old_ chunks?
                    let p = ivec3(x as i32, y as i32, z as i32) + dir;
                    let n = CHUNK_NUM as i32;

                    if p.x >= 0 && p.y >= 0 && p.z >= 0 && p.x < n && p.y < n && p.z < n {
                        // It's in bounds, we already have the blocks for it; just move the offset
                        *c = self.chunks.chunks[p.z as usize][p.y as usize][p.x as usize];
                        // assert!(self.origin.z != 112 || p.z != 0);
                    } else {
                        // if self.origin.z == 112 {
                        //     print!("Old: {:?}, new: ", p);
                        // }
                        // It's out of bounds, we need to make a new chunk and delete the old one

                        // Wrap the coordinates around. If it's `-1`, this will be `CHUNK_NUM-1`;
                        //  if it's `CHUNK_NUM`, this will be `CHUNK_NUM % CHUNK_NUM = 0`.
                        //  And if it's something else, it won't change
                        let p = (p + n) % n;
                        // if self.origin.z == 112 {
                        //     println!("{:?}", p);
                        // }
                        // A now-unoccupied chunk
                        let i = self.chunks.chunks[p.z as usize][p.y as usize][p.x as usize];
                        *c = i;

                        // Generate a new chunk and add it to `blocks`
                        let i = (i.2 as usize, i.1 as usize, i.0 as usize);
                        // let y = if self.origin.z == 112 {
                        //     0
                        // } else { y };
                        let new_chunk = self.gen.gen_chunk(
                            // World-space chunk coordinates, in chunks instead of blocks
                            ivec3(start.z,start.y,start.x) + ivec3(z as i32, y as i32, x as i32),
                        );
                        {
                            self.pix_buf.write(
                                &new_chunk
                                    .iter()
                                    .flat_map(|x| x.iter())
                                    .flat_map(|x| x.iter())
                                    .cloned()
                                    .collect::<Vec<Block>>(),
                            );
                            let i = (i.0 as u32, i.1 as u32, i.2 as u32);
                            let s = CHUNK_SIZE as u32;
                            self.block_buf.main_level().raw_upload_from_pixel_buffer(
                                self.pix_buf.as_slice(),
                                i.0..i.0 + s,
                                i.1..i.1 + s,
                                i.2..i.2 + s,
                            );
                        }
                        for (z, page) in new_chunk.iter().enumerate() {
                            for (y, row) in page.iter().enumerate() {
                                for (x, b) in row.iter().enumerate() {
                                    self.chunks.blocks[i.2 + z][i.1 + y][i.0 + x] = *b;
                                }
                            }
                        }
                    }
                }
            }
        }
        self.dpix_buf.write(
            &new_chunks
                .iter()
                .flat_map(|x| x.iter())
                .flat_map(|x| x.iter())
                .cloned()
                .collect::<Vec<(u8, u8, u8)>>(),
        );
        self.chunk_buf.main_level().raw_upload_from_pixel_buffer(
            self.dpix_buf.as_slice(),
            0..CHUNK_NUM as u32,
            0..CHUNK_NUM as u32,
            0..CHUNK_NUM as u32,
        );
        self.chunks.chunks = new_chunks;
    }

    /// Loads in new chunks if necessary, given the player position
    pub fn update(&mut self, player: Vec3) {
        let diff = player - to_vec3(self.origin);
        let t = CHUNK_SIZE as f32 * 0.5;
        // Has the player gone more than half a chunk away from the origin (ie, left the chunk)?
        if abs(diff.x) > t || abs(diff.y) > t || abs(diff.z) > t {
            let dir = if abs(diff.x) > abs(diff.y) {
                if abs(diff.x) > abs(diff.z) {
                    ivec3(sign(diff.x) as i32, 0, 0)
                } else {
                    ivec3(0, 0, sign(diff.z) as i32)
                }
            } else {
                if abs(diff.y) > abs(diff.z) {
                    ivec3(0, sign(diff.y) as i32, 0)
                } else {
                    ivec3(0, 0, sign(diff.z) as i32)
                }
            };
            // println!("Loading new chunk in direction {:?}", dir);
            self.load(dir);
        }
    }
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
