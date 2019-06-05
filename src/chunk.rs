use glm::*;
use glium::*;
use std::collections::HashMap;
use crate::common::*;
use crate::terrain::*;
use std::sync::mpsc::*;

pub struct ChunkThread {
    pub gen: Gen,
    ch: (Sender<ChunkMessage>, Receiver<ChunkMessage>),
}

impl ChunkThread {
    pub fn new(to: Sender<ChunkMessage>, from: Receiver<ChunkMessage>) -> Self {
        ChunkThread {
            gen: Gen::new(),
            ch: (to, from),
        }
    }

    pub fn run(self) {
        loop {
            match self.ch.1.recv() {
                Ok(ChunkMessage::LoadChunks(chunks)) => {
                    let ret = chunks.into_iter().map(|x|
                        (x,self.gen.gen(x))
                    ).collect();

                    self.ch.0.send(ChunkMessage::Chunks(ret)).unwrap();
                },
                Ok(ChunkMessage::UnloadChunk(_,_)) => {
                    // TODO save chunk
                },
                _ => break,
            }
        }
    }
}
