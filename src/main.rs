#[macro_use]
extern crate glium;
extern crate glsl_include;
pub extern crate nalgebra;
extern crate num_derive;
extern crate num_traits;

use glium::glutin;
use std::sync::Arc;

mod chunk;
mod chunk_thread;
mod client;
mod client_aux;
mod common;
mod config;
mod input;
mod material;
mod mesh;
mod physics;
mod server;
mod terrain;
mod world;

use client::*;
use common::*;
use server::*;

use std::fs::File;
use std::io::Write;

pub const APP_INFO: app_dirs2::AppInfo = app_dirs2::AppInfo {
    name: "voxrs",
    author: "Lorxu",
};

fn main() {
    // Wayland doesn't allow cursor grabbing
    let events_loop: glutin::EventsLoop = glutin::os::unix::EventsLoopExt::new_x11().unwrap();
    let wb = glutin::WindowBuilder::new()
        .with_title("Vox.rs")
        .with_fullscreen(Some(events_loop.get_primary_monitor()));
    let cb = glutin::ContextBuilder::new().with_depth_buffer(24);
    let display = glium::Display::new(wb, cb, &events_loop).unwrap();
    display.gl_window().window().grab_cursor(true).unwrap();
    display.gl_window().window().hide_cursor(true);

    let mut config_file =
        app_dirs2::app_root(app_dirs2::AppDataType::UserConfig, &APP_INFO).unwrap();
    config_file.push("config.ron");
    let client_config = if config_file.exists() {
        ron::de::from_reader(File::open(config_file).unwrap()).expect("bad config file")
    } else {
        let c = ClientConfig {
            mesher: crate::mesh::Mesher::Greedy,
            wireframe: false,
            batch_size: 16,
            keycodes: crate::input::DEFAULT_KEY_CODES,
            game_config: Arc::new(GameConfig {
                draw_chunks: 16,
                batch_size: 64,
                save_chunks: true,
            }),
        };
        let s = ron::ser::to_string(&c).unwrap();
        let mut f = File::create(config_file).unwrap();
        writeln!(f, "{}", s).unwrap();
        c
    };
    let client_config = Arc::new(client_config);

    let config = Arc::clone(&client_config.game_config);

    let resolution: (u32, u32) = display
        .gl_window()
        .window()
        .get_inner_size()
        .unwrap()
        .into();

    let camera_pos = Vec3::new(4.0, 16.0, 4.0);

    let (conn_client, conn_server) = Connection::local();
    let client = Client::new(display, Arc::clone(&client_config), conn_client, camera_pos);
    std::thread::spawn(move || {
        let mut server = Server::new(config);
        server.join(conn_server, camera_pos);
        server.run();
    });

    client.game_loop(resolution, events_loop);
}
