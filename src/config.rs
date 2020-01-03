use crate::chunk::*;
use std::sync::Arc;
use crate::mesh::Mesher;
use serde::{Deserialize, Serialize};


/// Config for both the client and server
#[derive(Deserialize, Serialize)]
pub struct GameConfig {
    pub draw_chunks: usize, // The number of chunks to draw in every direction
    pub batch_size: usize, // The number of chunks to load per batch
}

/// Config for just the client
#[derive(Deserialize, Serialize)]
pub struct ClientConfig {
    pub mesher: Mesher,
    pub wireframe: bool,
    pub batch_size: usize, // The number of chunks to mesh per batch

    pub keycodes: crate::input::KeyCodes,

    pub game_config: Arc<GameConfig>,
}
