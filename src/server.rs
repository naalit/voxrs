use glm::*;
use glium::backend::Facade;
use super::chunk::*;
use super::common::*;
use std::collections::HashMap;
use std::sync::mpsc::*;
use std::sync::Arc;
use super::client::*;

// #[derive(Clone)]
pub struct PlayerData {
    pos: Vec3,
    chunks: Vec<Vec<Vec<(u8, u8, u8)>>>,
    channel: (Sender<CommandList>,Receiver<Vec3>),
}

pub struct Server {
    chunk_map: HashMap<[i32; 3], Chunk>,
    ref_map: HashMap<[i32; 3], u32>,
    players: Vec<PlayerData>,
    chunk_man: (Sender<Message>,Receiver<Message>),
}

impl Server {
    pub fn new() -> Self {
        let chunk_man = ChunkManager::new();
        let (them_to,from) = channel();
        let (to,them_from) = channel();
        std::thread::spawn(move || chunk_man.chunk_thread(them_to, them_from));
        Server {
            chunk_map: HashMap::new(),
            ref_map: HashMap::new(),
            players: Vec::new(),
            chunk_man: (to, from),
        }
    }

    pub fn add_player(&mut self, pos: Vec3) -> Client {
        let chunk_pos = chunk(pos);
        let start = chunk_pos - CHUNK_NUM as i32 / 2;
        let chunk_locs = (0..CHUNK_NUM as i32)
            .flat_map(|x| (0..CHUNK_NUM as i32).map(move |y| ivec2(x,y)))
            .flat_map(|xy| (0..CHUNK_NUM as i32).map(move |z| ivec3(xy.x,xy.y,z)))
            .map(|x| start + x)
            .filter(|x| !self.chunk_map.contains_key(x.as_array()))
            .collect();
        self.chunk_man.0.send(Message::LoadChunks(chunk_locs)).unwrap();
        let chunks = self.chunk_man.1.recv().unwrap();
        if let Message::Chunks(chunks) = chunks {
            for (l,c) in chunks {
                self.chunk_map.insert(*l.as_array(),c);
                self.ref_map.insert(*l.as_array(),0);
            }
        }
        let mut chunks = Chunks::new();
        let mut i = 0;
        for (z, page) in chunks.map.iter_mut().enumerate() {
            for (y, row) in page.iter_mut().enumerate() {
                for (x, n) in row.iter_mut().enumerate() {
                    let p = start + ivec3(x as i32, y as i32, z as i32);
                    chunks.chunks[i] = self.chunk_map[p.as_array()];
                    *self.ref_map.get_mut(p.as_array()).unwrap() += 1;
                    *n = i;
                    i += 1;
                }
            }
        }
        let chunks = chunks.to_uniform();
        let (to,from_them) = channel();
        let (to_them,from) = channel();
        let p = PlayerData {
            pos,
            chunks: chunks.chunks.clone(),
            channel: (to,from),
        };
        self.players.push(p);
        Client::new(&chunks, pos, (to_them,from_them))
    }

    pub fn update(&mut self, old_pos: Vec3, new_pos: Vec3, player: &mut PlayerData) {
        let old_chunk = chunk(old_pos);
        let new_chunk = chunk(new_pos);
        if old_chunk != new_chunk {
            // let dir = ivec3( // TODO does this work for multiple directions at once (e.g. ivec3(1,1,0)?)
            //     sign(new_chunk.x-old_chunk.x),
            //     sign(new_chunk.y-old_chunk.y),
            //     sign(new_chunk.z-old_chunk.z)
            // );
            let dir = sign(new_chunk - old_chunk); // TODO does this work for multiple directions at once (e.g. ivec3(1,1,0)?)
            let mut chunks_load = Vec::new();
            // let mut chunks_unload = Vec::new();
            let mut new_chunks = player.chunks.clone();
            // Advance the origin
            // self.origin = self.origin + dir * CHUNK_SIZE as i32;
            // World-space chunk coordinates, in chunks instead of blocks
            let start = new_chunk - CHUNK_NUM as i32 / 2;
            let start = ivec3(start.z,start.y,start.x);
            let start_old = old_chunk - CHUNK_NUM as i32 / 2;
            let start_old = ivec3(start_old.z,start_old.y,start_old.x);


            (0..CHUNK_NUM as i32)
                .flat_map(|x| (0..CHUNK_NUM as i32).map(move |y| ivec2(x,y)))
                .flat_map(|xy| (0..CHUNK_NUM as i32).map(move |z| ivec3(xy.x,xy.y,z)))
                .map(|x| start_old + x)
                .filter(|x| { let d = *x - start; d.min() < 0 || d.max() >= CHUNK_NUM as i32 } )
                .for_each(|x| {
                    *self.ref_map.get_mut(x.as_array()).unwrap() -= 1;
                });

            for (z, page) in new_chunks.iter_mut().enumerate() {
                for (y, row) in page.iter_mut().enumerate() {
                    for (x, c) in row.iter_mut().enumerate() {
                        // Where would this chunk be in the _old_ chunks?
                        let p = ivec3(x as i32, y as i32, z as i32) + dir;
                        let n = CHUNK_NUM as i32;

                        if p.x >= 0 && p.y >= 0 && p.z >= 0 && p.x < n && p.y < n && p.z < n {
                            // It's in bounds, we already have the blocks for it; just move the offset
                            *c = player.chunks[p.z as usize][p.y as usize][p.x as usize];
                        } else {
                            // It's out of bounds, we need to make a new chunk and delete the old one

                            // Wrap the coordinates around. If it's `-1`, this will be `CHUNK_NUM-1`;
                            //  if it's `CHUNK_NUM`, this will be `CHUNK_NUM % CHUNK_NUM = 0`.
                            //  And if it's something else, it won't change
                            let p = (p + n) % n;

                            // A now-unoccupied chunk
                            let i = player.chunks[p.z as usize][p.y as usize][p.x as usize];
                            *c = i;

                            // Generate a new chunk and add it to `blocks`
                            let i = (i.2 as usize, i.1 as usize, i.0 as usize);
                            // let y = if self.origin.z == 112 {
                            //     0
                            // } else { y };
                            let new_chunk =
                                // World-space chunk coordinates, in chunks instead of blocks
                                start + ivec3(z as i32, y as i32, x as i32);
                            chunks_load.push((new_chunk,(i.0 as u32, i.1 as u32, i.2 as u32)));
                        }
                    }
                }
            }

            self.chunk_man.0.send(Message::LoadChunks(chunks_load.iter().map(|x|x.0).collect())).unwrap();
            let chunks = self.chunk_man.1.recv().unwrap();
            if let Message::Chunks(chunks) = chunks {
                for (l,c) in chunks {
                    self.chunk_map.insert(*l.as_array(),c);
                    self.ref_map.insert(*l.as_array(),0);
                }
            }

            let chunks_load = chunks_load.into_iter().map(|(l,o)| {
                *self.ref_map.get_mut(l.as_array()).unwrap() += 1;
                (self.chunk_map[l.as_array()].clone(),o)
            }).collect();

            // let z = Arc::from(new_chunks.clone());
            player.chunks = new_chunks.clone();
            player.channel.0.send(CommandList(
                chunks_load,
                new_chunks,
                new_chunk,
            )).unwrap();
        }
    }

    pub fn start(&mut self) {
        loop {
            self.tick();
        }
    }

    pub fn tick(&mut self) {
        let mut p = Vec::new();
        std::mem::swap(&mut p, &mut self.players);
        for i in p.iter_mut() {
            let mut p = i.pos;
            while let Ok(x) = i.channel.1.try_recv() {
                p = x;
            }
            self.update(i.pos,p,i);
            i.pos = p;
        }
        self.players = p;
        // TODO physics, etc.

        // Unload unneeded chunks
        let mut chunks_unload = Vec::new();
        for (l,r) in self.ref_map.iter() {
            if *r <= 0 {
                if let Some((l,c)) = self.chunk_map.remove_entry(l) {
                    chunks_unload.push((*IVec3::from_array(&l),c));
                }
            }
        }
        // We removed them from the chunk_map, now remove them from the ref_map
        self.ref_map.retain(|_k,v| *v > 0);
        if !chunks_unload.is_empty() {
            self.chunk_man.0.send(Message::UnloadChunks(chunks_unload)).unwrap();
        }
    }
}
