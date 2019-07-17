use std::sync::Arc;
use crate::mesh::Mesher;

/// Config for both the client and server
pub struct GameConfig {
    pub draw_chunks: usize, // The number of chunks to draw in every direction
}

/// Config for just the client
pub struct ClientConfig {
    pub mesher: Box<Mesher>,
    pub wireframe: bool,
    pub game_config: Arc<GameConfig>,
}
