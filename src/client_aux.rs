// This is the client auxilary thread, which is in charge of recieving chunks, meshing them, and sending them to the client thread.

use crate::common::*;
use crate::mesh::Vertex;
use std::collections::HashMap;
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
    mut player: Vec3,
    config: Arc<ClientConfig>,
) {
    // This is a timer for sending player movement to the server. We don't want to do it too often, just around 20 times per second.
    // So, we only send it when this timer is past 50ms
    let mut timer = stopwatch::Stopwatch::start_new();

    let mut chunk_map: HashMap<IVec3, Arc<RwLock<Chunk>>> = HashMap::new();
    let mut indices: Vec<IVec3> = Vec::new();
    let mut counter = 0;

    loop {
        if let Ok(m) = client.1.try_recv() {
            if let Message::PlayerMove(p) = m {
                player = p;
                if timer.elapsed_ms() > 50 {
                    server.send(m).expect("Disconnected from server");
                    timer.restart();
                }
            }
        } else {
            // Sync up with the client; we don't want to send more than one batch per frame
            continue;
        }
        if indices.len() != 0 {
            let meshed = indices
                .drain(0..8.min(indices.len()))
                .map(|loc| (loc,chunk_map.remove(&loc).unwrap()))
                .map(|(loc, chunk)| (loc, config.mesher.mesh(&chunk.read().unwrap()), Arc::clone(&chunk)))
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
            counter += 1;

            if counter >= 10 && indices.len() != 0 {
                println!("Rechunking");
                indices.sort_by_key(|x| ((chunk_to_world(*x) - player).norm() * 10.0) as i32);
                while indices.len() != 0 && (chunk_to_world(*indices.last().unwrap()) - player).norm() > DRAW_DIST {
                    let r = indices.pop().unwrap();
                    chunk_map.remove(&r);
                }

                counter = 0;
            }
        }
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
                    chunk_map.extend(chunks.into_iter().map(|(x,y)| (x,Arc::new(RwLock::new(y))) ));
                    indices = chunk_map.keys().cloned().collect();
                    indices.sort_by_key(|x| ((chunk_to_world(*x) - player).norm() * 10.0) as i32);
                    while indices.len() != 0 && (chunk_to_world(*indices.last().unwrap()) - player).norm() > DRAW_DIST {
                        let r = indices.pop().unwrap();
                        chunk_map.remove(&r);
                    }
                    counter = 0;
                }
                _ => (),
            }
        }
    }
}
