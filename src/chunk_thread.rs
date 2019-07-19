use crate::common::*;
use crate::terrain::*;
use std::sync::mpsc::*;
use std::sync::Arc;

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
                let ret: Vec<_> = to_load.drain(0..self.config.batch_size.min(to_load.len())).map(|x|
                    (x,self.gen.gen(x))
                ).collect();

                println!("Loaded {} chunks", ret.len());

                self.ch.0.send(ChunkMessage::Chunks(ret)).unwrap();

                match self.ch.1.try_recv() {
                    Ok(ChunkMessage::LoadChunks(mut chunks)) => {
                        to_load.append(&mut chunks);
                    },
                    Ok(ChunkMessage::UnloadChunk(_,_)) => {
                        // TODO save chunk
                    },
                    Err(TryRecvError::Disconnected) => break,
                    _ => {},
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
                    _ => break,
                }
            }

        }
    }
}
