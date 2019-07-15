// This is the client auxilary thread, which is in charge of recieving chunks, meshing them, and sending them to the client thread.

use crate::common::*;
use crate::mesh::Vertex;
use crate::mesh::*;
use std::sync::mpsc::*;
use std::sync::{Arc, RwLock};

pub type ClientMessage = Vec<(
    IVec3,
    Vec<Vertex>,
    Option<nc::shape::ShapeHandle<f32>>,
    Arc<RwLock<Chunk>>,
)>;

pub fn client_aux_thread(
    server: Connection,
    client: (Sender<ClientMessage>, Receiver<Message>),
    player: Vec3,
    config: Arc<ClientConfig>,
) {
    // This is a timer for sending player movement to the server. We don't want to do it too often, just around 20 times per second.
    // So, we only send it when this timer is past 50ms
    let mut timer = stopwatch::Stopwatch::start_new();

    loop {
        if let Some(m) = server.recv() {
            match m {
                Message::Chunks(chunks) => {
                    /*
                    println!(
                        "Requested load of {} chunks: \n{:?}",
                        chunks.len(),
                        chunks.iter().map(|x| x.0).collect::<Vec<IVec3>>()
                    );
                    */
                    let meshed = chunks
                        .into_iter()
                        .map(|(loc, chunk)| (loc, config.mesher.mesh(&chunk), Arc::new(RwLock::new(chunk))))
                        .map(|(loc, mesh, chunk)| {
                            if mesh.len() != 0 {
                                let v_physics: Vec<_> =
                                    mesh.iter().map(|x| na::Point3::from(x.pos)).collect();
                                let i_physics: Vec<_> = (0..v_physics.len() / 3)
                                    .map(|x| na::Point3::new(x * 3, x * 3 + 1, x * 3 + 2))
                                    .collect();
                                let chunk_shape = nc::shape::ShapeHandle::new(
                                    nc::shape::TriMesh::new(v_physics, i_physics, None),
                                );
                                (loc, mesh, Some(chunk_shape), chunk)
                            } else {
                                (loc, mesh, None, chunk)
                            }
                        })
                        .collect();
                    client.0.send(meshed).unwrap();
                }
                _ => (),
            }
        }
        if let Ok(m) = client.1.try_recv() {
            if let Message::PlayerMove(p) = m {
                if timer.elapsed_ms() > 50 {
                    server.send(m).expect("Disconnected from server");
                    timer.restart();
                }
            }
        }
    }
}
