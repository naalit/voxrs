use crate::common::*;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

pub struct World {
    pub chunks: HashMap<IVec3, Chunk>,
}

pub type ArcWorld = Arc<RwLock<World>>;
pub fn arcworld() -> ArcWorld {
    Arc::new(RwLock::new(World::new()))
}

impl World {
    pub fn new() -> Self {
        World {
            chunks: HashMap::new(),
        }
    }

    pub fn locs(&self) -> std::collections::hash_map::Keys<'_, IVec3, Chunk> {
        self.chunks.keys()
    }

    pub fn chunk(&self, k: &IVec3) -> Option<&Chunk> {
        self.chunks.get(k)
    }
    pub fn add_chunk(&mut self, k: IVec3, v: Chunk) {
        self.chunks.insert(k, v);
    }
    pub fn remove_chunk(&mut self, k: &IVec3) -> Option<Chunk> {
        self.chunks.remove(k)
    }
    pub fn block(&self, k: Vec3) -> Option<Material> {
        let chunk = world_to_chunk(k);
        let in_chunk = in_chunk(k);
        let chunk = self.chunks.get(&chunk)?;
        Some(chunk.block(in_chunk))
    }
    pub fn set_block(&mut self, k: Vec3, v: Material) {
        let chunk = world_to_chunk(k);
        let in_chunk = in_chunk(k);
        let chunk = self.chunks.get_mut(&chunk).unwrap();
        chunk.set_block(in_chunk, v);
    }
}

impl Extend<(IVec3, Chunk)> for World {
    fn extend<T: IntoIterator<Item = (IVec3, Chunk)>> (&mut self, it: T) {
        self.chunks.extend(it);
    }
}
