/// #GUIDE TO TYPES
/// In order to not confuse different spaces, we always use the same types for spaces:
/// - `Vec3` is a position in world-space, in increments of 1 meter
/// - `IVec3` is a chunk location in world-space, in increments of 1 chunk

pub use glm::*;
use std::sync::mpsc::*;

// MUST BE power of 2
pub const CHUNK_NUM: UVec3 = Vector3 { x: 4, y: 4, z: 4 };
pub const CHUNK_NUM_I: IVec3 = IVec3 { x: CHUNK_NUM.x as i32 / 2, y: CHUNK_NUM.y as i32 / 2, z: CHUNK_NUM.z as i32 / 2 };

pub const CHUNK_SIZE: f32 = 16.0;
pub const DRAW_DIST: f32 = CHUNK_SIZE * 2.0;

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


pub type Material = u16;
pub const AIR: Material = 0;
pub type Chunk = Vec<Vec<Vec<Material>>>;

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
