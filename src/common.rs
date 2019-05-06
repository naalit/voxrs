pub use glm::*;
use std::sync::Arc;

// Should be a power of 2
pub const CHUNK_SIZE: usize = 16;
// This is a 'diameter'
pub const CHUNK_NUM: usize = 16;

pub type Block = u16;
pub type Chunk = [[[Block; CHUNK_SIZE]; CHUNK_SIZE]; CHUNK_SIZE];

pub enum Message {
    LoadChunks(Vec<IVec3>),
    UnloadChunks(Vec<(IVec3,Chunk)>),
    Chunks(Vec<(IVec3,Chunk)>),
}

pub fn chunk(pos: Vec3) -> IVec3 {
    to_ivec3(pos) / CHUNK_SIZE as i32
}

#[derive(Clone)]
pub struct CommandList (
    pub Vec<(Chunk, (u32,u32,u32))>,
    pub Vec<Vec<Vec<(u8, u8, u8)>>>,
    pub IVec3,
);
