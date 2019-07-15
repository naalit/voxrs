#[macro_use]
extern crate glium;
extern crate glsl_include;
pub extern crate nalgebra;
extern crate num_traits;
#[macro_use]
extern crate num_derive;

use glium::glutin;
use glium::Surface;
use glsl_include::Context as ShaderContext;
use num_traits::identities::*;

mod chunk;
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
    let mut events_loop: glutin::EventsLoop = glutin::os::unix::EventsLoopExt::new_x11().unwrap();
    let wb = glutin::WindowBuilder::new()
        .with_title("Vox.rs 2")
        .with_fullscreen(Some(events_loop.get_primary_monitor()));
    let cb = glutin::ContextBuilder::new().with_depth_buffer(24);
    let display = glium::Display::new(wb, cb, &events_loop).unwrap();
    display.gl_window().window().grab_cursor(true).unwrap();
    display.gl_window().window().hide_cursor(true);

    let resolution: (u32, u32) = display
        .gl_window()
        .window()
        .get_inner_size()
        .unwrap()
        .into();

    let mut camera_pos = Vec3::new(4.0, 16.0, 4.0);

    let (conn_client, conn_server) = Connection::local();
    let mut client = Client::new(display, conn_client, camera_pos);
    std::thread::spawn(move || {
        let mut server = Server::new();
        server.join(conn_server, camera_pos);
        server.run();
    });

    client.game_loop(resolution, events_loop);
}
