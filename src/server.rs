use std::sync::Arc;
use crate::common::*;
use crate::chunk::*;
use std::collections::{HashSet, HashMap};
use std::sync::mpsc::*;
use std::thread;
use std::rc::Rc;

struct Player {
    pos: Vec3,
    conn: Rc<Connection>,
    id: usize,
    //to_send: Vec<IVec3>,
}

pub struct Server {
    chunks: HashMap<IVec3, Chunk>,
    refs: HashMap<IVec3, usize>,
    players: Vec<Player>,
    orders: HashMap<IVec3, Vec<(usize, Rc<Connection>)>>,
    ch: (Sender<ChunkMessage>, Receiver<ChunkMessage>),
    config: Arc<GameConfig>,
}

impl Server {
    /// Creates and starts a chunk thread, and creates a Server
    pub fn new(config: Arc<GameConfig>) -> Self {
        let (to, from_them) = channel();
        let (to_them, from) = channel();
        let c = Arc::clone(&config);
        thread::spawn(move ||{
            ChunkThread::new(c, to_them, from_them).run()
        });

        Server {
            chunks: HashMap::new(),
            refs: HashMap::new(),
            players: Vec::new(),
            orders: HashMap::new(),
            ch: (to, from),
            config,
        }
    }

    /// Add a player to the game
    pub fn join(&mut self, conn: Connection, pos: Vec3) {
        let new_player = Player {
            pos,
            conn: Rc::new(conn),
            id: self.players.len(),
        };
        let (wait, load) = self.load_chunks_around(pos);
        //p.to_send.append(&mut wait);
        for i in wait {
            self.orders.entry(i).or_insert_with(|| Vec::new()).push((new_player.id, Rc::clone(&new_player.conn)));
        }
        if load.len() > 0 {
            new_player.conn.send(Message::Chunks(load)).unwrap();
        }
        self.players.push(new_player);
    }

    /// Runs an infinite tick loop. It's infinite, start as a new thread!
    pub fn run(mut self) {
        let mut running = true;
        while running {
            let mut p = Vec::new();
            std::mem::swap(&mut p, &mut self.players);
            self.players = p.into_iter().filter_map(|mut p| {
                let mut np = p.pos;
                while let Some(m) = p.conn.recv() {
                    match m {
                        Message::PlayerMove(n_pos) => {
                            np = n_pos;
                        },
                        Message::Leave => match *p.conn {
                            Connection::Local(_,_) => { running = false; break; },
                            _ => return None,
                        },
                        _ => panic!("Hey, a client sent a message {:?}", m),
                    }
                }
                let (wait, load) = self.load_chunk_diff(p.pos, np);
                //p.to_send.append(&mut wait);
                for i in wait {
                    self.orders.entry(i).or_insert_with(|| Vec::new()).push((p.id, Rc::clone(&p.conn)));
                }
                if load.len() > 0 {
                    p.conn.send(Message::Chunks(load)).unwrap();
                }
                p.pos = np;
                Some(p)
            }).collect();

            while let Ok(m) = self.ch.1.try_recv() {
                match m {
                    ChunkMessage::Chunks(x) => {
                        let locs: Vec<IVec3> = x.iter().map(|y| y.0).collect();

                        let mut batches = HashMap::new();
                        for (i, c) in &x {
                            if let Some(v) = self.orders.remove(&i) {
                                for (id, conn) in v {
                                    batches.entry(id).or_insert_with(|| (conn, Vec::new())).1.push((*i, c.clone()));
                                }
                            }
                        }
                        for (_, (conn, v)) in batches {
                            conn.send(Message::Chunks(v));
                        }

                        for (loc, chunk) in x.into_iter() {
                            self.chunks.insert(loc, chunk); // There's a very small chance we're replacing a chunk; see Todo below
                        }
                    },
                    _ => panic!("Chunk thread sent {:?}", m),
                }
            }

        }
    }

    /// Loads initial chunks around a player
    /// Returns `(chunks_to_wait_for, chunks_already_loaded)`
    /// Doesn't update `orders`
    fn load_chunks_around(&mut self, pos: Vec3) -> (Vec<IVec3>, Vec<(IVec3, Chunk)>) {
        let chunk_pos = world_to_chunk(pos);

        let mut to_load = Vec::new();

        let draw_chunks = self.config.draw_chunks as i32;

        for x in -draw_chunks..draw_chunks {
            for y in -draw_chunks..draw_chunks {
                for z in -draw_chunks..draw_chunks {
                    let p = IVec3::new(x,y,z);
                    if p.map(|x|x as f32).norm() <= self.config.draw_chunks as f32 {
                        to_load.push(chunk_pos+p);
                    }
                }
            }
        }

        to_load.sort_by_cached_key(|a| ((a.map(|x|x as f32)).norm() * 10.0) as i32);

        let mut to_send = Vec::new();
        let mut to_pass = Vec::new();
        for p in to_load {
            match self.chunks.get(&p) {
                Some(chunk) => to_pass.push((p,chunk.clone())),
                None        => to_send.push(p),
            }
            match self.refs.get_mut(&p) {
                Some(x) => *x += 1, // This is indeed possible but ugly; see Todo below
                None    => { self.refs.insert(p,1); },
            }
        }

        // If it's already being loaded, don't tell the chunk thread to load it again.
        // The calling function will add this player to `orders` too, so we don't need to bother here
        to_send.retain(|x| !self.orders.contains_key(&x));

        self.ch.0.send(ChunkMessage::LoadChunks(to_send.clone())).unwrap();
        (to_send, to_pass)
    }

    /// Figures out what chunks need to be loaded, and either returns them or sends them to the chunk thread
    /// Returns `(chunks_to_wait_for, chunks_already_loaded)`
    /// Doesn't update `orders`
    fn load_chunk_diff(&mut self, old: Vec3, new: Vec3) -> (Vec<IVec3>, Vec<(IVec3, Chunk)>) {
        let chunk_old = world_to_chunk(old);
        let chunk_new = world_to_chunk(new);

        if chunk_old == chunk_new {
            return (Vec::new(), Vec::new());
        }

        println!("Loading chunks around {:?}", new);

        let mut around_old = HashSet::new();
        let mut around_new = HashSet::new();
        let draw_chunks = self.config.draw_chunks as i32;

        for x in -draw_chunks..draw_chunks {
            for y in -draw_chunks..draw_chunks {
                for z in -draw_chunks..draw_chunks {
                    let p = IVec3::new(x,y,z);
                    if p.map(|x|x as f32).norm() <= self.config.draw_chunks as f32 {
                        around_old.insert(chunk_old+p);
                        around_new.insert(chunk_new+p);
                    }
                }
            }
        }
        let to_load = &around_new - &around_old;
        let to_unload = &around_old - &around_new;

        for i in to_unload {
            if self.refs.contains_key(&i) {
                let r = {
                    // Lower the refcount on this chunk by one
                    let q = self.refs.get_mut(&i).expect("Tried to unload a chunk that isn't loaded");
                    let r = *q - 1;
                    *q = r;
                    r
                };
                // If the refcount is zero, nobody's using it so we can unload it
                if r <= 0 {
                    // TODO tell chunk thread to unload this chunk
                    self.chunks.remove(&i);
                    self.refs.remove(&i);
                }
            } else {
                panic!("Tried to unload a chunk that isn't loaded [2]: {:?}", i);
            }
        }

        let mut to_send = Vec::new();
        let mut to_pass = Vec::new();
        for p in to_load {
            match self.chunks.get(&p) {
                Some(chunk) => to_pass.push((p,chunk.clone())),
                None        => to_send.push(p),
            }
            match self.refs.get_mut(&p) {
                Some(x) => *x += 1, // This is indeed possible but ugly; see Todo below
                None    => { self.refs.insert(p,1); },
            }
        }

        // If it's already being loaded, don't tell the chunk thread to load it again.
        // The calling function will add this player to `orders` too, so we don't need to bother here
        to_send.retain(|x| !self.orders.contains_key(&x));

        to_send.sort_by_cached_key(|a| ((a - chunk_new).map(|x|x as f32).norm() * 10.0) as i32);

        self.ch.0.send(ChunkMessage::LoadChunks(to_send.clone())).unwrap();
        (to_send, to_pass)
    }
}
