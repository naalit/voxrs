use crate::common::*;
use crate::terrain::*;
use std::sync::mpsc::*;
use std::sync::Arc;
use stopwatch::Stopwatch;

pub struct ChunkThread {
    pub gen: Gen,
    ch: (Sender<ChunkMessage>, Receiver<ChunkMessage>),
    config: Arc<GameConfig>,
}

impl ChunkThread {
    pub fn new(config: Arc<GameConfig>, to: Sender<ChunkMessage>, from: Receiver<ChunkMessage>) -> Self {
        ChunkThread {
            gen: Gen::new(),
            ch: (to, from),
            config,
        }
    }

    pub fn run(self) {
        let mut to_load = Vec::new();

        loop {
            if to_load.len() != 0 {
                // let timer = Stopwatch::start_new();
                let ret: Vec<_> = to_load.drain(0..self.config.batch_size.min(to_load.len())).map(|x|
                    (x,self.gen.gen(x))
                ).collect();

                // let l = ret.len();
                // println!("Loaded {} chunks", l);

                self.ch.0.send(ChunkMessage::Chunks(ret)).unwrap();

                // println!("Loading took {} ms/chunk, {} ms total", timer.elapsed_ms() as f64 / l as f64, timer.elapsed_ms());

                let mut connected = true;
                let mut sort = Vec::new();
                loop {
                    match self.ch.1.try_recv() {
                        Ok(ChunkMessage::LoadChunks(mut chunks)) => {
                            to_load.append(&mut chunks);
                        },
                        Ok(ChunkMessage::UnloadChunk(_,_)) => {
                            // TODO save chunk
                        },
                        Ok(ChunkMessage::Players(players)) => {
                            sort = players;
                        },
                        Err(TryRecvError::Disconnected) => { connected = false; break; },
                        _ => break,
                    }
                }
                if !connected { break; }
                if sort.len() != 0 {
                    // let timer = Stopwatch::start_new();
                    to_load.retain(|x| sort.iter().any(|y| (world_to_chunk(*y) - x).map(|x| x as f32).norm() < self.config.draw_chunks as f32));
                    to_load.sort_by_cached_key(|x| { let x = chunk_to_world(*x); sort.iter().map(|y| ((x - y).norm() * 100.0) as usize).min() });
                    // println!("Sorting took {} ms for to_load len {}", timer.elapsed().as_micros() as f64 / 1000.0, to_load.len());
                }
            } else {
                // Wait for more chunks to load
                match self.ch.1.recv() {
                    Ok(ChunkMessage::LoadChunks(mut chunks)) => {
                        to_load.append(&mut chunks);
                    },
                    Ok(ChunkMessage::UnloadChunk(_,_)) => {
                        // TODO save chunk
                    },
                    Ok(ChunkMessage::Players(_)) => {},
                    _ => break,
                }
            }

        }
    }
}
