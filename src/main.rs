#[macro_use]
extern crate glium;
extern crate enum_iterator;
extern crate glm;
extern crate glsl_include;
extern crate noise;
extern crate num_traits;
extern crate rand;
extern crate rayon;
extern crate stopwatch;

mod chunk;
mod client;
mod common;
mod material;
mod server;
mod terrain;

use client::Client;
use common::*;
use server::Server;

fn main() {
    let mut server = Server::new();
    let mut client = server.add_player(vec3(0.0, 25.0, 72.0));
    std::thread::spawn(move || server.start());

    client.game_loop();
}
