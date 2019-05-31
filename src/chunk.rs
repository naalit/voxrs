use glm::*;
use glium::*;
use std::collections::HashMap;

pub const CHUNK_NUM: UVec3 = Vector3 { x: 2, y: 2, z: 2 };
pub const CHUNK_SIZE: f32 = 2.0;

#[derive(Copy, Clone, Debug)]
pub struct Node { pub pointer: [u32; 8] }
implement_uniform_block!(Node, pointer);

impl Node {
    /// An empty leaf node
    pub fn new() -> Self {
        Node { pointer: [0; 8] }
    }

    /// Converts between a 3D vector representing the child slot, and the actual index into the `pointer` array
    pub fn idx<T: BaseNum>(idx: Vector3<T>) -> usize {
        // Once again, this function closely mirrors the GLSL one for testing
        let mut ret = 0;
        ret |= usize::from(idx.x > T::zero()) << 2;
        ret |= usize::from(idx.y > T::zero()) << 1;
        ret |= usize::from(idx.z > T::zero());
        ret
    }

    /// Converts between a 3D vector representing the child slot, and the actual index into the `pointer` array
    pub fn position(idx: usize) -> Vector3<f32> {
        vec3(
            if idx & (1 << 2) > 0 { 1.0 } else { -1.0 },
            if idx & (1 << 1) > 0 { 1.0 } else { -1.0 },
            if idx & 1 > 0 { 1.0 } else { -1.0 },
        )
    }
}

pub fn as_tuple<T: BaseNum>(x: Vector3<T>) -> (T,T,T) {
    (x.x,x.y,x.z)
}

pub fn chunk_to_world(chunk: IVec3) -> Vec3 { (to_vec3(chunk) - 0.5) * CHUNK_SIZE }

pub type Chunk = Vec<Node>;

pub struct ClientData {
    origin: Vec3,
    player: Vec3,
    root_size: f32,
    root_max_level: u32,
    pub root: Vec<Node>, // The root structure. Points to chunks, gets buffer in the map
    chunks: HashMap<(i32,i32,i32),Chunk>,
    pub map: HashMap<(i32,i32,i32),(usize,usize)>, // (start, end)
    spaces: Vec<(usize,usize)>, // (start, end)
    tree_buffer: uniforms::UniformBuffer<[Node]>,
}

impl ClientData {
    /// TEMPORARY - TODO
    pub fn new(display: &Display) -> Self {
        let start_len = 240000;
        let mut max_root_size = CHUNK_NUM.x * CHUNK_NUM.y * CHUNK_NUM.z;
        let mut last = max_root_size * 8;
        while last > 0 {
            last = last / 8;
            max_root_size += last;
        }
        println!("Max root size = {}", max_root_size);
        ClientData {
            origin: vec3(0.0,0.0,0.0),
            player: vec3(0.0,0.0,0.0),
            root_size: CHUNK_NUM.max() as f32 * CHUNK_SIZE,
            root_max_level: 2,
            root: vec![Node::new()],
            chunks: HashMap::new(),
            map: HashMap::new(),
            spaces: vec![(max_root_size as usize,start_len)],
            tree_buffer: uniforms::UniformBuffer::empty_unsized(display, start_len * std::mem::size_of::<Node>()).unwrap(),
        }
    }

    /// Gets a reference to `tree_buffer`, for use as a uniform
    pub fn tree_uniform(&self) -> &uniforms::UniformBuffer<[Node]> {
        &self.tree_buffer
    }

    /// Load a bunch of chunks at once. Prunes the root as well, so it needs the player position
    /// Uploads everything to the GPU
    pub fn load_chunks(&mut self, player: Vec3, chunks: Vec<(IVec3, Chunk)>) {
        self.player = player;
        for (i,c) in chunks {
            self.load(i,c);
        }
        self.prune_root();
        self.upload_root();
    }

    pub fn upload_root(&mut self) {
        self.tree_buffer.slice_mut(0..self.root.len()).unwrap().write(self.root.as_slice());
    }

    /// Loads a chunk in at position `idx` in world-space (divided by CHUNK_SIZE)
    /// Will automatically unload the chunk that was previously there.
    /// Uploads this chunk to the GPU, but not the modified root structure.
    pub fn load(&mut self, idx: IVec3, chunk: Chunk) {
        // Unload the previous chunk
        self.unload(idx);

        // We need this much space
        // We add 64 to allow for the chunk to grow without moving. We'll move it if it goes past 32
        let size = chunk.len() + 64;

        // Find a space
        let mut i = 0;
        let (start,end) = loop {
            let (space_start, space_end) = self.spaces[i];
            let space_size = space_end - space_start;
            if space_size == size {
                // Our chunk fits EXACTLY, so just remove this space
                self.spaces.remove(i);
                break (space_start, space_end);
            }
            if space_size > size {
                // Our chunk fits, so we can shrink this space
                self.spaces[i] = (space_start+size, space_end);
                break (space_start, space_start+size);
            }

            // This one doesn't fit, so move on to the next space
            i += 1;
            if i >= self.spaces.len() {
                // We're to the end of `spaces`, so this chunk can't fit anywhere
                panic!("Could not find space for chunk {:?}, size {}!", idx, size);
            }
        };

        println!("Found a space at {}", start);

        // Add this chunk's pointer to the root structure
        let center = chunk_to_world(idx);
        self.add_to_root(center, start as u32);

        // Add the 64 empty nodes here
        let mut chunk_gpu = chunk.clone();
        chunk_gpu.append(&mut vec![Node::new(); 64]);

        // Add to map & chunks
        self.chunks.insert(as_tuple(idx), chunk);
        self.map.insert(as_tuple(idx), (start, end));

        // Upload to GPU
        self.tree_buffer.slice_mut(start..end).unwrap().write(chunk_gpu.as_slice());
    }

    /// Unload the chunk at position `idx` in world space.
    /// This is the client function, so it won't store it anywhere or anything, that's the server's job.
    pub fn unload(&mut self, idx: IVec3) {
        if let Some((start, end)) = self.map.remove(&as_tuple(idx)) {
            self.chunks.remove(&as_tuple(idx));

            // Add a space
            for i in 0..self.spaces.len() {
                let (space_start,space_end) = self.spaces[i];

                if space_start == end {
                    // This space was at the end of our chunk, so we can just extend it backwards to fill the space
                    self.spaces[i] = (start,space_end);
                    break;
                }
                if space_end == start {
                    // Our chunk was just after this space, so we can extend the space forwards
                    self.spaces[i] = (space_start, end);
                    break;
                }

                if space_start > end {
                    // This space is after our chunk, so we'll put our new space here. It's like insertion sort
                    self.spaces.insert(i,(start,end));
                    break;
                }

                // This space is before our chunk, so we'll keep going until we find the right position
            }

            // We don't have to touch GPU memory, because we aren't necessarily replacing this chunk with anything
        }
    }

    /// Adds a pointer to the root structure. The pointer should point to a chunk
    /// DOES NOT transfer root structure to GPU!
    fn add_to_root(&mut self, pos: Vec3, pointer: u32) {
        if distance(pos,self.origin) > self.root_size * 0.5 {
            // It's outside the root node, we need a new one

            let d = sign(pos-self.origin + 0.0001); // Add 0.0001 so we don't get 0s
            let idx = -d;
            self.origin = self.origin - d * 0.5 * self.root_size;
            self.root_size = self.root_size * 2.0;
            let uidx = Node::idx(idx);
            let mut new_root = Node::new();
            new_root.pointer[uidx] = 0b11; // The old root will be in index 1 relative to the new root, and it's nonempty

            self.root.insert(0,new_root);
        }

        let mut cur = self.origin;
        let mut size = self.root_size;
        let mut parent = self.root[0];
        let mut parent_pointer = 0;
        let mut idx = vec3(0.0,0.0,0.0);
        for i in 0.. {
            size *= 0.5;
            idx = sign(pos-cur+0.0001); // mix_bool(vec3(-1.0,-1.0,-1.0),vec3(1.0,1.0,1.0),greaterThan(pos,cur));
            cur = cur + idx * size * 0.5;

            if size > CHUNK_SIZE && i < self.root_max_level {
                // Descend again

                let uidx = Node::idx(idx);
                let node = parent.pointer[uidx];
                if node & 1 == 0 {
                    // Doesn't go further, we need to add it ourselves

                    let n_pointer = self.root.len() as u32;
                    self.root[parent_pointer as usize].pointer[uidx] = ((n_pointer - parent_pointer) << 1) | 1;
                    parent = Node::new(); // An empty leaf node as a placeholder
                    self.root.push(parent.clone());
                    parent_pointer = n_pointer;

                } else {
                    parent_pointer = parent_pointer + node >> 1;
                    parent = self.root[parent_pointer as usize];
                }
            } else { break; }
        }

        self.root[parent_pointer as usize].pointer[Node::idx(idx)] = ((pointer - parent_pointer) << 1) | 1;
    }

    /// Removes unnecessary (too far away) nodes from root structure
    /// DOES NOT transfer root structure to the GPU!
    fn prune_root(&mut self) {
        // The root should never be empty!
        assert!(self.prune_root_node(self.origin, self.root_size, 0));

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
        }

    }

    /// Returns whether to keep this node (assumes it's not the root, so one child still matters)
    fn prune_root_node(&mut self, pos: Vec3, size: f32, pointer: usize) -> bool {
        let node = self.root[pointer];
        let mut count = 0;
        let size = size * 0.5; // Size of children
        for i in 0..8 {
            let np = pos + Node::position(i) * size * 0.5;

            // Are we in range of the player?
            if distance(np,self.player) <= self.root_size {
                if node.pointer[i] != 0 {
                    count += 1;
                }
            } else if node.pointer[i] != 0 {
                // Remove node and all brethren
                self.remove_root_node_recursive(node.pointer[i]);
            }
        }
        count > 0
    }

    /// Remove a node & shift necessary pointers
    fn remove_root_node(&mut self, pointer: u32) -> Node {
        // Shift pointers
        self.root.iter_mut().take(pointer as usize) // All the nodes before this one
            .for_each(|x| for i in 0..8 { if x.pointer[i] > pointer { x.pointer[i] -= 0b10 } }); // Only subtract if it's ahead of `pointer`.
            // If `x[i] == pointer`, this function SHOULD NOT HAVE BEEN CALLED!

        // Actually remove it
        self.root.remove(pointer as usize)
    }

    /// `remove_root_node`, but it removes all children recursively as well
    fn remove_root_node_recursive(&mut self, pointer: u32) {
        let node = self.remove_root_node(pointer);
        for i in 0..8 {
            if node.pointer[i] & 1 > 0 {
                let p = pointer + (node.pointer[i] >> 1);
                if p < self.root.len() as u32 { // Don't try to remove something that's not in the root
                    self.remove_root_node_recursive(p);
                }
            }
        }
    }
}
