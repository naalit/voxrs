use crate::common::*;
use glium::*;
use std::collections::HashMap;

pub struct Client {
    conn: Connection,
    origin: Vec3,
    player: Vec3,
    pub root_size: f32,
    pub root: Vec<Node>, // The root structure. Points to chunks, gets buffer in the map
    chunks: HashMap<(i32, i32, i32), Chunk>,
    pub map: HashMap<(i32, i32, i32), (usize, usize)>, // (start, end)
    spaces: Vec<(usize, usize)>,                       // (start, end)
    tree_buffer: uniforms::UniformBuffer<[Node]>,
}

impl Client {
    pub fn new(display: &Display, conn: Connection, player: Vec3) -> Self {
        let start_len = 3_200_000; // * 32 bytes = 100 MB
        let mut max_root_size = CHUNK_NUM.x * CHUNK_NUM.y * CHUNK_NUM.z;
        let mut last = max_root_size * 8;
        while last > 0 {
            last = last / 8;
            max_root_size += last;
        }
        println!("Max root size = {}", max_root_size);
        Client {
            conn,
            origin: fmod(player, vec3(CHUNK_SIZE,CHUNK_SIZE,CHUNK_SIZE)),
            player,
            root_size: 8.0,//CHUNK_NUM.max() as f32 * CHUNK_SIZE,
            root: vec![Node::new()],
            chunks: HashMap::new(),
            map: HashMap::new(),
            spaces: vec![(max_root_size as usize, start_len)],
            tree_buffer: uniforms::UniformBuffer::empty_unsized(
                display,
                start_len * std::mem::size_of::<Node>(),
            )
            .unwrap(),
        }
    }

    /// The player position
    pub fn pos(&self) -> Vec3 {
        self.player
    }

    /// Gets a reference to `tree_buffer`, for use as a uniform
    pub fn tree_uniform(&self) -> &uniforms::UniformBuffer<[Node]> {
        &self.tree_buffer
    }

    pub fn origin_uniform(&self) -> [f32; 3] {
        *self.origin.as_array()
    }

    pub fn update(&mut self, player: Vec3) {
        self.player = player;
        if let Some(m) = self.conn.recv() {
            // Only load chunks once per frame
            match m {
                Message::Chunks(chunks) => {
                    /*
                    println!(
                        "Requested load of {} chunks: \n{:?}",
                        chunks.len(),
                        chunks.iter().map(|x| x.0).collect::<Vec<IVec3>>()
                    );
                    */
                    self.load_chunks(chunks)
                }
                _ => (),
            }
        }
        self.conn.send(Message::PlayerMove(player));
    }

    /// Load a bunch of chunks at once. Prunes the root as well
    /// Uploads everything to the GPU
    pub fn load_chunks(&mut self, chunks: Vec<(IVec3, Chunk)>) {
        for (i, c) in chunks {
            self.load(i, c);
        }

        self.prune_chunks();
        self.create_root();
        self.upload_root();
    }

    pub fn upload_root(&mut self) {
        self.tree_buffer
            .slice_mut(0..self.root.len())
            .unwrap()
            .write(self.root.as_slice());
    }

    /// Loads a chunk in at position `idx` in world-space (divided by CHUNK_SIZE)
    /// Will automatically unload the chunk that was previously there.
    /// Uploads this chunk to the GPU, but not the modified root structure.
    pub fn load(&mut self, idx: IVec3, chunk: Chunk) {
        // Unload the previous chunk
        self.unload(idx);

        // We need this much space
        // We add 64 to allow for the chunk to grow without moving. We'll move it if it goes past 32 - TODO
        let size = chunk.len() + 64;

        // Find a space
        let mut i = 0;
        let (start, end) = loop {
            let (space_start, space_end) = self.spaces[i];
            let space_size = space_end - space_start;
            if space_size == size {
                // Our chunk fits EXACTLY, so just remove this space
                self.spaces.remove(i);
                break (space_start, space_end);
            }
            if space_size > size {
                // Our chunk fits, so we can shrink this space
                self.spaces[i] = (space_start + size, space_end);
                break (space_start, space_start + size);
            }

            // This one doesn't fit, so move on to the next space
            i += 1;
            if i >= self.spaces.len() {
                // We're to the end of `spaces`, so this chunk can't fit anywhere
                panic!("Could not find space for chunk {:?}, size {}!", idx, size);
            }
        };

        // println!("Found a space at {}", start);

        // Add the 64 empty nodes here
        let mut chunk_gpu = chunk.clone();
        chunk_gpu.append(&mut vec![Node::new(); 64]);

        // Add to map & chunks
        self.chunks.insert(as_tuple(idx), chunk);
        self.map.insert(as_tuple(idx), (start, end));

        // Upload to GPU
        self.tree_buffer
            .slice_mut(start..end)
            .unwrap()
            .write(chunk_gpu.as_slice());
    }

    /// Unload the chunk at position `idx` in world space.
    /// This is the client function, so it won't store it anywhere or anything, that's the server's job.
    pub fn unload(&mut self, idx: IVec3) {
        if let Some((start, end)) = self.map.remove(&as_tuple(idx)) {
            self.chunks.remove(&as_tuple(idx));

            // Add a space
            for i in 0..self.spaces.len() {
                let (space_start, space_end) = self.spaces[i];

                if space_start == end {
                    // This space was at the end of our chunk, so we can just extend it backwards to fill the space
                    self.spaces[i] = (start, space_end);
                    break;
                }
                if space_end == start {
                    // Our chunk was just after this space, so we can extend the space forwards
                    self.spaces[i] = (space_start, end);
                    break;
                }

                if space_start > end {
                    // This space is after our chunk, so we'll put our new space here. It's like insertion sort
                    self.spaces.insert(i, (start, end));
                    break;
                }

                // This space is before our chunk, so we'll keep going until we find the right position
            }

            // We don't have to touch GPU memory, because we aren't necessarily replacing this chunk with anything
        }
    }

    /// Unloads chunks that are too far away
    fn prune_chunks(&mut self) {
        for i in self.map.clone().keys() {
            let i = as_vec(*i);
            let p = chunk_to_world(i);
            let d = length(p - self.player);
            if d > ROOT_SIZE {
                self.unload(i);
            }
        }
    }

    /// Recreates the root node to incorporate newly loaded chunks
    fn create_root(&mut self) {
        // Find the extent of the root in each direction
        let mut h = ivec3(-1000000,-1000000,-1000000);
        let mut l = ivec3(1000000,1000000,1000000);
        for i in self.chunks.keys() {
            if i.0 > h.x { h.x = i.0; }
            if i.1 > h.y { h.y = i.1; }
            if i.2 > h.z { h.z = i.2; }

            if i.0 < l.x { l.x = i.0; }
            if i.1 < l.y { l.y = i.1; }
            if i.2 < l.z { l.z = i.2; }
        }

        let h = chunk_to_world(h);
        let l = chunk_to_world(l);

        self.origin = chunk_to_world(world_to_chunk((h+l)*0.5)) + CHUNK_SIZE * 0.5;
        self.root_size = abs(h-l).max() + CHUNK_SIZE; // Add two halves of a chunk
        self.root_size = exp2(ceil(log2(self.root_size))); // Round up to a power of 2

        self.root = self.create_node(self.origin, self.root_size, 0);
    }

    /// Create a node in the root structure, returning that node and all children
    fn create_node(&self, pos: Vec3, size: f32, pointer: usize) -> Vec<Node> {
        let size = size * 0.5; // Child size
        let mut ret = Vec::new();
        ret.push(Node::new()); // ret[0] is the node we're actually working on
        for uidx in 0..8 {
            let idx = Node::position(uidx);
            let pos = pos + idx * size * 0.5;
            if size > CHUNK_SIZE {
                // Descend
                let ptr = ret.len(); // Relative pointer to the first of the new nodes
                ret.append(&mut self.create_node(pos, size, pointer+ptr));
                let ptr = (ptr << 1) | 1;
                ret[0].pointer[uidx] = ptr as u32;
            } else {
                // This is a chunk, so figure out which one
                let chunk_loc = world_to_chunk(pos);
                let ptr = if let Some((chunk_ptr, _)) = self.map.get(&as_tuple(chunk_loc)) {
                    ((chunk_ptr - pointer) << 1) | 1
                } else {
                    // There's no chunk here, it's empty
                    0
                };
                ret[0].pointer[uidx] = ptr as u32;
            }
        }
        ret
    }

    /* /// This stuff is all deprecated. It modifies the root instead of creating a new one every time chunks are loaded, and it doesn't work very well.

    /// Adds a pointer to the root structure. The pointer should point to a chunk
    /// DOES NOT transfer root structure to GPU!
    fn add_to_root(&mut self, pos: Vec3, new_pointer: u32) {
        println!("Adding pos ({}, {}, {}) to root size {} around ({}, {}, {})", pos.x, pos.y, pos.z, self.root_size, self.origin.x, self.origin.y, self.origin.z);

        if abs(pos-self.origin).max() + CHUNK_SIZE * 0.5 > self.root_size * 0.5 {
            // We need a new root

            println!("Creating new root");

            // The idx of the old root in the new one
            let idx = sign(self.origin - pos + 0.0001); // Add 0.0001 so we don't get 0s
            self.origin = self.origin - idx * 0.5 * self.root_size;
            self.root_size *= 2.0;
            let uidx = Node::idx(idx);
            let mut new_root = Node::new();
            new_root.pointer[uidx] = 0b11; // The old root will be at position 1

            // Shift chunk pointers
            let l = self.root.len();
            self.root
                .iter_mut()
                .enumerate() // We need the indices for relative pointers
                .for_each(|(n, x)| {
                    for i in 0..8 {
                        let p = n as u32 + (x.pointer[i] >> 1);
                        if x.pointer[i] & 1 > 0 && p > l as u32 {
                            // It's not a leaf, and it's not in the root
                            x.pointer[i] -= 0b10
                            // Shift it backwards, because the chunks didn't move forwards with the root nodes
                        }
                    }
                });

            self.root.insert(0, new_root);
        }

        let mut size = self.root_size;
        let mut cur = self.origin;
        let mut idx;
        let mut parent_pointer = 0;
        let mut parent = self.root[0];

        loop {
            size *= 0.5;
            idx = sign(pos - cur + 0.0001);
            cur = cur + idx * size * 0.5;

            if size > CHUNK_SIZE {
                // Descend again
                let uidx = Node::idx(idx);
                let node = parent.pointer[uidx];
                if node & 1 == 0 {
                    // Doesn't go further, we need to add it ourselves

                    println!("Adding new child");

                    let n_pointer = self.root.len() as u32;
                    self.root[parent_pointer as usize].pointer[uidx] =
                        ((n_pointer - parent_pointer) << 1) | 1;
                    parent = Node::new(); // An empty leaf node as a placeholder
                    self.root.push(parent.clone());
                    parent_pointer = n_pointer;

                } else {
                    parent_pointer += node >> 1;
                    parent = self.root[parent_pointer as usize];
                }
            } else {
                break;
            }
        }

        self.root[parent_pointer as usize].pointer[Node::idx(idx)] =
            ((new_pointer - parent_pointer) << 1) | 1;
    }

    /// Removes unnecessary (too far away) nodes from root structure
    /// DOES NOT transfer root structure to the GPU!
    fn prune_root(&mut self) {
        // The root should never be empty!
        assert!
        (self.prune_root_node(self.origin, self.root_size, 0));

        let node = self.root[0];
        let mut count = 0;
        let mut last = 0;
        for i in 0..8 {
            if node.pointer[i] != 0 && node.pointer[i] >> 1 < self.root.len() as u32 {
                // A child, in the root structure
                count += 1;
                last = i;
            }
        }
        if count == 1 {
            println!("Replacing root with single child...");

            // Since the root has only one child, and nodes can only refer to others later in the tree,
            //  the root's one child should be at position 1, and we don't need to change any pointers.
            self.root.remove(0);

            // We do need to change the origin and size, though.
            self.root_size = self.root_size * 0.5;
            self.origin = self.origin + Node::position(last) * 0.5 * self.root_size;

            // Shift chunk pointers since root nodes moved
            let l = self.root.len();
            self.root
                .iter_mut()
                .enumerate() // We need the indices for relative pointers
                .for_each(|(n, x)| {
                    for i in 0..8 {
                        let p = n as u32 + (x.pointer[i] >> 1);
                        if x.pointer[i] & 1 > 0 && p > l as u32 {
                            // It's not a leaf, and it's not in the root
                            x.pointer[i] += 0b10
                            // Shift it forward, because the chunks didn't move backwards with the root nodes
                        }
                    }
                });
        }

    }

    /// Returns whether to keep this node (assumes it's not the root, so one child still matters)
    fn prune_root_node(&mut self, pos: Vec3, size: f32, pointer: usize) -> bool {
        let d_corner = 0.75_f32.sqrt();
        let mut node = self.root[pointer];
        let mut count = 0;
        let size = size * 0.5; // Size of children
        for i in 0..8 {
            let np = pos + Node::position(i) * size * 0.5;

            // Is ANY of this node range of the player?
            if distance(np, self.player)-size*d_corner <= 0.5 * ROOT_SIZE {
                if node.pointer[i] != 0 {
                    let p = pointer + (node.pointer[i] as usize >> 1);
                    if node.pointer[i] & 1 > 0 && p < self.root.len() {
                        // It's not a leaf, and it's in the root
                        self.prune_root_node(np, size, p);
                    }
                    count += 1;
                }
            } else if node.pointer[i] & 1 > 0 {
                // Remove node and all children
                self.remove_root_node_recursive(pointer as u32 + (node.pointer[i] >> 1));
                self.root[pointer].pointer[i] = 0; // It's empty now
                node = self.root[pointer]; // Reload, the pointers have changed
            }
        }
        count > 0
    }

    /// Remove a node & shift necessary pointers
    fn remove_root_node(&mut self, pointer: u32) -> Node {
        // Shift pointers
        let l = self.root.len();
        self.root
            .iter_mut()
            .take(pointer as usize) // All the nodes before this one
            .enumerate() // We need the indices for relative pointers
            .for_each(|(n, x)| {
                for i in 0..8 {
                    let p = n as u32 + (x.pointer[i] >> 1);
                    if x.pointer[i] & 1 > 0 && p > pointer && p < l as u32 {
                        // It's not a leaf, it's to after `pointer`, and it's in the root, so it moved back one
                        x.pointer[i] -= 0b10
                    }
                }
            });

        // Shift chunk pointers too
        self.root
            .iter_mut()
            .enumerate() // We need the indices for relative pointers
            .skip(pointer as usize + 1) // All the nodes that got moved
            .for_each(|(n, x)| {
                for i in 0..8 {
                    let p = n as u32 + (x.pointer[i] >> 1);
                    if x.pointer[i] & 1 > 0 && p > l as u32 {
                        // It's not a leaf, and it's not in the root
                        x.pointer[i] += 0b10
                        // Shift it forward, because the chunks didn't move backwards with the root nodes
                    }
                }
            });

        // Actually remove it
        self.root.remove(pointer as usize)
    }

    /// `remove_root_node`, but it removes all children recursively as well
    fn remove_root_node_recursive(&mut self, pointer: u32) {
        if pointer < self.root.len() as u32 {
            let mut node = self.root[pointer as usize];
            for i in 0..8 {
                if node.pointer[i] & 1 > 0 {
                    let p = pointer + (node.pointer[i] >> 1);
                    /*let p = if p <= self.root.len() as u32 {
                        // Subtract 1 since we just removed the node at `pointer`, everything after is shifted left
                        p - 1
                    } else {
                        // But chunks didn't move
                        p
                    };*/
                    self.remove_root_node_recursive(p);
                    node = self.root[pointer as usize]; // The recursion might have changed the child pointers
                }
            }
            self.remove_root_node(pointer);
        } else {
            // It's a chunk, find and remove it
            self.unload(*self.rev_map.get(&(pointer as usize)).expect(&format!(
                "Tried to unload nonexistent chunk at index {}",
                pointer
            )));
        }
    }
    */
}
