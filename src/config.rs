use crate::chunk::*;
use std::sync::Arc;
use crate::mesh::Mesher;

/// Config for both the client and server
pub struct GameConfig {
    pub draw_chunks: usize, // The number of chunks to draw in every direction
    pub batch_size: usize, // The number of chunks to load per batch
}

/// Config for just the client
pub struct ClientConfig {
    pub mesher: Box<Mesher>,
    pub wireframe: bool,
    pub batch_size: usize, // The number of chunks to mesh per batch

    pub game_config: Arc<GameConfig>,
}
