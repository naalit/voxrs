#[macro_use]
extern crate glium;
extern crate glm;
extern crate glsl_include;
extern crate rayon;
extern crate stopwatch;
extern crate noise;
extern crate rand;

mod chunk;
mod terrain;
mod common;
mod server;
mod client;

use common::*;
use server::Server;
use client::Client;

fn main() {
    let mut server = Server::new();
    let mut client = server.add_player(vec3(0.0,25.0,72.0));
    std::thread::spawn(move ||
        server.start());

    client.game_loop();
}
