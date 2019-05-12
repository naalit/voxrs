pub use glm::*;
use std::sync::Arc;

// Should be a power of 2
pub const CHUNK_SIZE: usize = 16;
// This is a 'diameter'
pub const CHUNK_NUM: usize = 24;

pub type Block = u16;
pub type Chunk = Option<[[[Block; CHUNK_SIZE]; CHUNK_SIZE]; CHUNK_SIZE]>;

pub enum Message {
    LoadChunks(Vec<IVec3>),
    UnloadChunks(Vec<(IVec3, Chunk)>),
    Chunks(Vec<(IVec3, Chunk)>),
}

pub fn chunk(pos: Vec3) -> IVec3 {
    ivec3(
        if pos.x < 0.0 { -1 } else { 0 }, // x=2 is chunk x=0, but x=-2 is chunk x=-1
        if pos.y < 0.0 { -1 } else { 0 },
        if pos.z < 0.0 { -1 } else { 0 },
    ) + to_ivec3(pos) / CHUNK_SIZE as i32
}

pub fn in_chunk(pos: Vec3) -> IVec3 {
    let chunk = chunk(pos);
    let chunk_start = chunk * CHUNK_SIZE as i32;
    (to_ivec3(pos) - chunk_start + CHUNK_SIZE as i32) % CHUNK_SIZE as i32
}

#[derive(Clone)]
pub struct CommandList(
    pub Vec<(Chunk, (u32, u32, u32))>,
    pub Vec<Vec<Vec<(u8, u8, u8)>>>,
    pub IVec3,
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_pos() {
        let p = vec3(-2.0, 3.0, 25.2);
        assert_eq!(chunk(p), ivec3(-1, 0, 1));
        assert_eq!(in_chunk(p), ivec3(14, 3, 9));

        let p = vec3(-22.0, 12.4, 1.2);
        assert_eq!(chunk(p), ivec3(-2, 0, 0));
        assert_eq!(in_chunk(p), ivec3(10, 12, 1));
    }
}
