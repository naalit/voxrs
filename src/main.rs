#[macro_use]
extern crate glium;
extern crate glsl_include;
pub extern crate nalgebra;
extern crate num_traits;
extern crate num_derive;

use glium::glutin;
use std::sync::Arc;

mod chunk;
mod chunk_thread;
mod client;
mod client_aux;
mod config;
mod common;
mod input;
mod material;
mod mesh;
mod server;
mod terrain;

use client::*;
use common::*;
use server::*;

fn main() {
    // Wayland doesn't allow cursor grabbing
    let events_loop: glutin::EventsLoop = glutin::os::unix::EventsLoopExt::new_x11().unwrap();
    let wb = glutin::WindowBuilder::new()
        .with_title("Vox.rs 2")
        .with_fullscreen(Some(events_loop.get_primary_monitor()));
    let cb = glutin::ContextBuilder::new().with_depth_buffer(24);
    let display = glium::Display::new(wb, cb, &events_loop).unwrap();
    display.gl_window().window().grab_cursor(true).unwrap();
    display.gl_window().window().hide_cursor(true);

    let config = GameConfig {
        draw_chunks: 16,
        batch_size: 64,
    };
    let config = Arc::new(config);

    let resolution: (u32, u32) = display
        .gl_window()
        .window()
        .get_inner_size()
        .unwrap()
        .into();

    let camera_pos = Vec3::new(4.0, 16.0, 4.0);

    let (conn_client, conn_server) = Connection::local();
    let client = Client::new(display, Arc::clone(&config), conn_client, camera_pos);
    std::thread::spawn(move || {
        let mut server = Server::new(config);
        server.join(conn_server, camera_pos);
        server.run();
    });

    client.game_loop(resolution, events_loop);
}
