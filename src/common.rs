// Should be a power of 2
pub const CHUNK_SIZE: usize = 16;
// This is a 'diameter'
pub const CHUNK_NUM: usize = 16;

pub type Block = u16;
pub type Chunk = [[[Block; CHUNK_SIZE]; CHUNK_SIZE]; CHUNK_SIZE];
