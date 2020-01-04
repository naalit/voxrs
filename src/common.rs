/// #GUIDE TO TYPES
/// In order to not confuse different spaces, we always use the same types for coordinates:
/// - `Vec3` is a position in world-space, in increments of 1 meter
/// - `IVec3` is a chunk location in world-space, in increments of 1 chunk
/// - `UVec3` is a block within a chunk
pub use nalgebra as na;
pub use nalgebra::{Isometry3, Point3, Scalar, Unit, Vector3};
pub use ncollide3d as nc;
pub use np::object::Body;
pub use nphysics3d as np;
use std::sync::mpsc::*;

pub use crate::config::*;

pub const CHUNK_SIZE: f32 = 32.0;

// Shorthands to match GLSL
pub type IVec3 = Vector3<i32>;
pub type UVec3 = Vector3<usize>;
pub type Vec3 = Vector3<f32>;

pub fn radians(degrees: f32) -> f32 {
    std::f32::consts::PI / 180.0 * degrees
}

pub fn chunk_to_world(chunk: IVec3) -> Vec3 {
    chunk.map(|x| x as f32 + 0.5) * CHUNK_SIZE
}
pub fn world_to_chunk(world: Vec3) -> IVec3 {
    world.map(|x| x as i32 + if x < 0.0 { 1 } else { 0 }) / CHUNK_SIZE as i32
        - world.map(|x| if x < 0.0 { 1 } else { 0 })
}
/// The index of a block within its home chunk
pub fn in_chunk(world: Vec3) -> UVec3 {
    world.map(|x| {
        ((x as i32 % CHUNK_SIZE as i32) + CHUNK_SIZE as i32) as usize % CHUNK_SIZE as usize
    })
}

pub use crate::chunk::*;
pub use crate::material::*;

pub enum Connection {
    Local(Sender<Message>, Receiver<Message>),
    // TODO some sort of buffered TCP stream inplementation of Connection
}

impl Connection {
    /// Create a two new Local connections - (client, server)
    pub fn local() -> (Connection, Connection) {
        let (cto, sfrom) = channel();
        let (sto, cfrom) = channel();
        let client = Connection::Local(cto, cfrom);
        let server = Connection::Local(sto, sfrom);
        (client, server)
    }

    /// Equivalent to Sender::send() but as an option
    pub fn send(&self, m: Message) -> Option<()> {
        match self {
            Connection::Local(to, _from) => to.send(m).ok(),
        }
    }

    /// Equivalent to Receiver::try_recv() but as an option - doesn't block
    pub fn recv(&self) -> Option<Message> {
        match self {
            Connection::Local(_to, from) => from.try_recv().ok(),
        }
    }
}

#[derive(Debug)]
pub enum Message {
    PlayerMove(Vec3),
    Chunks(Vec<(IVec3, Chunk)>),
    SetBlock(IVec3, Material),
    Leave,
}

#[derive(Debug)]
pub enum ChunkMessage {
    Done,
    LoadChunks(Vec<IVec3>),
    // Chunks(Vec<(IVec3, Chunk)>),
    UnloadChunk(IVec3, Chunk),
    Players(Vec<Vec3>),
}
