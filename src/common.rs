/// #GUIDE TO TYPES
/// In order to not confuse different spaces, we always use the same types for spaces:
/// - `Vec3` is a position in world-space, in increments of 1 meter
/// - `IVec3` is a chunk location in world-space, in increments of 1 chunk

pub use glm::*;
use std::sync::mpsc::*;

// MUST BE power of 2
pub const CHUNK_NUM: UVec3 = Vector3 { x: 8, y: 8, z: 8 };
pub const CHUNK_NUM_I: IVec3 = IVec3 { x: CHUNK_NUM.x as i32 / 2, y: CHUNK_NUM.y as i32 / 2, z: CHUNK_NUM.z as i32 / 2 };

pub const CHUNK_SIZE: f32 = 2.0;
pub const ROOT_SIZE: f32 = 2.0 * 8.0;

#[derive(Copy, Clone, Debug)]
pub struct Node {
    pub pointer: [u32; 8],
}
implement_uniform_block!(Node, pointer);

impl Node {
    /// An empty leaf node
    pub fn new() -> Self {
        Node { pointer: [0; 8] }
    }

    /// Converts between a 3D vector representing the child slot, and the actual index into the `pointer` array
    pub fn idx<T: BaseNum>(idx: Vector3<T>) -> usize {
        // Once again, this function closely mirrors the GLSL one for testing
        let mut ret = 0;
        ret |= usize::from(idx.x > T::zero()) << 2;
        ret |= usize::from(idx.y > T::zero()) << 1;
        ret |= usize::from(idx.z > T::zero());
        ret
    }

    /// Converts between a 3D vector representing the child slot, and the actual index into the `pointer` array
    pub fn position(idx: usize) -> Vector3<f32> {
        vec3(
            if idx & (1 << 2) > 0 { 1.0 } else { -1.0 },
            if idx & (1 << 1) > 0 { 1.0 } else { -1.0 },
            if idx & 1 > 0 { 1.0 } else { -1.0 },
        )
    }
}

pub fn as_tuple<T: BaseNum>(x: Vector3<T>) -> (T, T, T) {
    (x.x, x.y, x.z)
}
pub fn as_vec<T: BaseNum>(x: (T,T,T)) -> Vector3<T> {
    Vector3 {
        x: x.0,
        y: x.1,
        z: x.2,
    }
}

pub fn chunk_to_world(chunk: IVec3) -> Vec3 {
    (to_vec3(chunk) - 0.5) * CHUNK_SIZE
}
pub fn world_to_chunk(world: Vec3) -> IVec3 {
    to_ivec3(world / CHUNK_SIZE + 0.5)
}

pub type Chunk = Vec<Node>;

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
    Leave,
}

#[derive(Debug)]
pub enum ChunkMessage {
    LoadChunks(Vec<IVec3>),
    Chunks(Vec<(IVec3, Chunk)>),
    UnloadChunk(IVec3, Chunk),
}
