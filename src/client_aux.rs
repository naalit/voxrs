// This is the client auxilary thread, which is in charge of recieving chunks, meshing them, and sending them to the client thread.

use crate::common::*;
use crate::mesh::Vertex;
use crate::mesh::*;
use std::sync::mpsc::*;
use std::sync::{Arc, RwLock};

pub type ClientMessage = Vec<(IVec3, Vec<Vertex>, Arc<RwLock<Chunk>>)>;

pub fn client_aux_thread(
    server: Connection,
    client: (Sender<ClientMessage>, Receiver<Message>),
    player: Vec3,
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
                        .map(|(loc, chunk)| (loc, mesh(&chunk), Arc::new(RwLock::new(chunk))))
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
