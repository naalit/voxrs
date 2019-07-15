use std::sync::Arc;
use crate::mesh::Mesher;

/// Config for both the client and server
pub struct GameConfig {

}

/// Config for just the client
pub struct ClientConfig {
    pub mesher: Box<Mesher>,
    pub wireframe: bool,
    pub game_config: GameConfig,
}
