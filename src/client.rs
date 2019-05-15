use super::chunk::*;
use super::common::*;
use super::material::*;
use glium::*;
use glsl_include::Context as ShaderContext;
use num_traits::identities::*;
use std::collections::HashMap;
use std::sync::mpsc::*;

struct FrameState {
    fb: f32,   // * camera_dir
    lr: f32,   // * right
    ud: f32,   // * vec3(0,1,0)
    m: f32,    // Multiplier for movement speed
    jump: f32, // Jump counter
    fly: bool,
    try_jump: bool,
}
impl FrameState {
    fn new() -> Self {
        FrameState {
            fb: 0.0,
            lr: 0.0,
            ud: 0.0,
            m: 1.0,
            jump: 0.0,
            fly: false,
            try_jump: false,
        }
    }
}

#[derive(Copy, Clone)]
struct Vertex {
    pos: [f32; 2],
}
implement_vertex!(Vertex, pos);

fn vert(x: f32, y: f32) -> Vertex {
    Vertex { pos: [x, y] }
}

fn shader(path: String, includes: &[String]) -> String {
    use std::fs::File;
    use std::io::Read;
    let mut file = File::open("src/".to_owned() + &path).unwrap();
    let mut string = String::new();
    file.read_to_string(&mut string).unwrap();
    let mut c = ShaderContext::new();
    for i in includes {
        let mut file = File::open("src/".to_owned() + &i).unwrap();
        let mut string = String::new();
        file.read_to_string(&mut string).unwrap();
        c.include(i.clone(), string);
    }
    c.expand(string).unwrap()
}

const MOVE_SPEED: f32 = 0.01;

pub struct Client {
    /// These two are the buffers for storing a chunk and the chunk map, respectively, to update them
    pix_buf: glium::texture::pixel_buffer::PixelBuffer<Block>,
    dpix_buf: glium::texture::pixel_buffer::PixelBuffer<(u8, u8, u8)>,
    /// The actual textures
    chunk_buf: glium::texture::unsigned_texture3d::UnsignedTexture3d,
    block_buf: glium::texture::unsigned_texture3d::UnsignedTexture3d,
    /// Local storage of chunks for player physics
    map: Vec<Vec<Vec<(u8, u8, u8)>>>,
    chunks: HashMap<(u8, u8, u8), Chunk>,

    /// We only keep these since we need to initialize them in the `Client::new`
    events_loop: Option<glutin::EventsLoop>,
    display: Display,

    state: FrameState,
    pos: Vec3,
    origin: IVec3,
    channel: (Sender<Message>, Receiver<Message>),
}

impl Client {
    pub fn new(
        chunks: &ChunksU,
        chunks_new: HashMap<(u8, u8, u8), Chunk>,
        pos: Vec3,
        channel: (Sender<Message>, Receiver<Message>),
    ) -> Self {
        let events_loop = if cfg!(target_os = "linux") {
            glutin::os::unix::EventsLoopExt::new_x11().unwrap()
        } else {
            glutin::EventsLoop::new()
        };
        let wb = glutin::WindowBuilder::new()
            .with_title("Vox.rs")
            .with_fullscreen(Some(events_loop.get_primary_monitor()))
            .with_decorations(false);
        let cb = glutin::ContextBuilder::new(); //.with_vsync(true);
        let display = glium::Display::new(wb, cb, &events_loop).unwrap();
        display.gl_window().window().grab_cursor(true).unwrap();
        display.gl_window().window().hide_cursor(true); //.unwrap();
                                                        // .set_cursor(glutin::MouseCursor::Crosshair); //.unwrap();

        let origin = chunk(pos);
        let block_buf = glium::texture::unsigned_texture3d::UnsignedTexture3d::with_format(
            &display,
            chunks.blocks.clone(),
            glium::texture::UncompressedUintFormat::U16,
            glium::texture::MipmapsOption::NoMipmap,
        )
        .unwrap();
        let chunk_buf = glium::texture::unsigned_texture3d::UnsignedTexture3d::with_format(
            &display,
            chunks.chunks.clone(),
            glium::texture::UncompressedUintFormat::U8U8U8,
            glium::texture::MipmapsOption::NoMipmap,
        )
        .unwrap();
        let map = chunks.chunks.clone();

        Client {
            pix_buf: glium::texture::pixel_buffer::PixelBuffer::new_empty(
                &display,
                CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE,
            ),
            dpix_buf: glium::texture::pixel_buffer::PixelBuffer::new_empty(
                &display,
                CHUNK_NUM * CHUNK_NUM * CHUNK_NUM,
            ),
            chunk_buf,
            block_buf,

            map,
            chunks: chunks_new,

            events_loop: Some(events_loop),
            display,

            state: FrameState::new(),
            pos,
            origin,
            channel,
        }
    }

    pub fn get_blocki(&self, pos: IVec3) -> u16 {
        self.get_block(to_vec3(pos) + 0.5)
    }

    pub fn get_block(&self, pos: Vec3) -> u16 {
        let chunk = chunk(pos) - self.origin + CHUNK_NUM as i32 / 2;
        let in_chunk = in_chunk(pos);
        let offset = self.map[chunk.z as usize][chunk.y as usize][chunk.x as usize];
        self.chunks[&offset].map_or(0, |x| {
            x[in_chunk.z as usize][in_chunk.y as usize][in_chunk.x as usize]
        })
    }

    /// `pos` is in local coordinates
    pub fn set_block(&mut self, pos: IVec3, b: u16) {
        let ichunk = pos / CHUNK_SIZE as i32;
        let in_chunk = pos % CHUNK_SIZE as i32;
        let offset = self.map[ichunk.z as usize][ichunk.y as usize][ichunk.x as usize];
        let chunk = self.chunks[&offset];
        let mut chunk = chunk.unwrap_or_else(|| [[[0; CHUNK_SIZE]; CHUNK_SIZE]; CHUNK_SIZE]);
        chunk[in_chunk.z as usize][in_chunk.y as usize][in_chunk.x as usize] = b;

        self.chunks.insert(offset, Some(chunk));

        self.pix_buf.write(
            &chunk
                .iter()
                .flat_map(|x| x.iter())
                .flat_map(|x| x.iter())
                .cloned()
                .collect::<Vec<Block>>(),
        );
        let s = CHUNK_SIZE as u32;
        let i = (offset.2 as u32, offset.1 as u32, offset.0 as u32);
        let i = (i.0 * s, i.1 * s, i.2 * s);
        self.block_buf.main_level().raw_upload_from_pixel_buffer(
            self.pix_buf.as_slice(),
            i.0..i.0 + s,
            i.1..i.1 + s,
            i.2..i.2 + s,
        );
        self.channel
            .0
            .send(Message::SetBlock(
                pos + self.origin * CHUNK_SIZE as i32 - CHUNK_SIZE as i32 * CHUNK_NUM as i32 / 2,
                b,
            ))
            .unwrap();
    }

    pub fn trace(&self, ro: Vec3, rd: Vec3, max_iter: i32) -> Option<IVec3> {
        self.trace_n(ro, rd, max_iter).map(|x| x.0)
    }

    /// Also returns the normal - `(position,normal)`
    pub fn trace_n(&self, ro: Vec3, rd: Vec3, max_iter: i32) -> Option<(IVec3, IVec3)> {
        let total_size = CHUNK_NUM * CHUNK_SIZE;
        let total_size = total_size as i32;
        let ro_chunk = ro - to_vec3(self.origin) * CHUNK_SIZE as f32 + total_size as f32 * 0.5;

        let mut ipos = to_ivec3(floor(ro_chunk));
        // let mut ipos = to_ivec3(ro) - self.origin * CHUNK_SIZE as i32 + total_size / 2;
        let tdelta = abs(Vec3::one() / rd);
        let istep = to_ivec3(sign(rd));
        let mut side_dist = (sign(rd) * (floor(ro) - ro) + (sign(rd) * 0.5) + 0.5) * tdelta;
        for _ in 0..max_iter {
            if any(lessThan(ipos, IVec3::zero()))
                || any(greaterThanEqual(
                    ipos,
                    ivec3(total_size, total_size, total_size),
                ))
            {
                return None;
            }
            let chunk = ipos / CHUNK_SIZE as i32;
            let offset = self.map[chunk.z as usize][chunk.y as usize][chunk.x as usize];
            if let Some(c) = self.chunks[&offset] {
                let in_chunk = ipos % CHUNK_SIZE as i32;
                let b = c[in_chunk.z as usize][in_chunk.y as usize][in_chunk.x as usize];
                if b != 0 {
                    let t = if side_dist.x < side_dist.y {
                        if side_dist.x < side_dist.z {
                            side_dist.x - tdelta.x
                        } else {
                            side_dist.z - tdelta.z
                        }
                    } else {
                        if side_dist.y < side_dist.z {
                            side_dist.y - tdelta.y
                        } else {
                            side_dist.z - tdelta.z
                        }
                    };
                    let p = ro_chunk + rd * t;
                    let n = p - to_vec3(ipos);
                    let n = to_ivec3(sign(n))
                        * if abs(n.x) > abs(n.y) {
                            // Not y
                            if abs(n.x) > abs(n.z) {
                                ivec3(1, 0, 0)
                            } else {
                                ivec3(0, 0, 1)
                            }
                        } else {
                            if abs(n.y) > abs(n.z) {
                                ivec3(0, 1, 0)
                            } else {
                                ivec3(0, 0, 1)
                            }
                        };
                    return Some((ipos, n));
                }
            } else {
                while ipos / CHUNK_SIZE as i32 == chunk {
                    if side_dist.x < side_dist.y {
                        if side_dist.x < side_dist.z {
                            side_dist.x += tdelta.x;
                            ipos.x += istep.x;
                        } else {
                            side_dist.z += tdelta.z;
                            ipos.z += istep.z;
                        }
                    } else {
                        if side_dist.y < side_dist.z {
                            side_dist.y += tdelta.y;
                            ipos.y += istep.y;
                        } else {
                            side_dist.z += tdelta.z;
                            ipos.z += istep.z;
                        }
                    }
                }
            }

            if side_dist.x < side_dist.y {
                if side_dist.x < side_dist.z {
                    side_dist.x += tdelta.x;
                    ipos.x += istep.x;
                } else {
                    side_dist.z += tdelta.z;
                    ipos.z += istep.z;
                }
            } else {
                if side_dist.y < side_dist.z {
                    side_dist.y += tdelta.y;
                    ipos.y += istep.y;
                } else {
                    side_dist.z += tdelta.z;
                    ipos.z += istep.z;
                }
            }
        }
        None
    }

    /// In-place moves ourself as far as we can, up to `d`, in normalized direction `dir`
    pub fn do_move(&mut self, dir: Vec3, d: f32) {
        if d == 0.0 {
            return;
        };

        let i = [self.pos];
        let i = i
            .iter()
            .flat_map(|x| vec![*x + vec3(0.0, 0.5, 0.0), *x - vec3(0.0, 1.5, 0.0)])
            .flat_map(|x| vec![x + vec3(0.4, 0.0, 0.0), x - vec3(0.4, 0.0, 0.0)])
            .flat_map(|x| vec![x + vec3(0.0, 0.0, 0.4), x - vec3(0.0, 0.0, 0.4)])
            .map(|x| to_ivec3(x))
            .unique_by(|x| *x.as_array());
        let i = i.collect::<Vec<IVec3>>();

        // for x in i.iter() {
        //     if self.get_blocki(*x) != 0 {
        //         println!("Error: player inside block!");
        //     }
        // }

        let full_pos = self.pos + dir * d;

        let i2 = [full_pos];
        let i2 = i2
            .iter()
            .flat_map(|x| vec![*x + vec3(0.0, 0.5, 0.0), *x - vec3(0.0, 1.5, 0.0)])
            .flat_map(|x| vec![x + vec3(0.4, 0.0, 0.0), x - vec3(0.4, 0.0, 0.0)])
            .flat_map(|x| vec![x + vec3(0.0, 0.0, 0.4), x - vec3(0.0, 0.0, 0.4)])
            .map(|x| to_ivec3(x))
            .unique_by(|x| *x.as_array())
            .filter(|x| !i.contains(x));

        if i2.map(|x| self.get_blocki(x)).all(|x| x == 0) {
            self.pos = full_pos;
        } else {
            let dir = dir * sign(d);
            let c = ceil(self.pos) - vec3(0.4, 0.5, 0.4); // Our size relative to the camera
            let f = floor(self.pos) + vec3(0.4, 0.5, 0.4);
            self.pos = vec3(
                if dir.x > c.x - self.pos.x {
                    c.x
                } else if dir.x < f.x - self.pos.x {
                    f.x
                } else {
                    self.pos.x
                },
                if dir.y > c.y - self.pos.x {
                    c.y
                } else if dir.y < f.y - self.pos.y {
                    f.y
                } else {
                    self.pos.y
                },
                if dir.z > c.z - self.pos.x {
                    c.z
                } else if dir.z < f.z - self.pos.z {
                    f.z
                } else {
                    self.pos.z
                },
            )
        }
    }

    pub fn can_move(&self, new_pos: Vec3) -> bool {
        [new_pos]
            .iter()
            .flat_map(|x| vec![*x + vec3(0.0, 0.5, 0.0), *x - vec3(0.0, 1.5, 0.0)])
            .flat_map(|x| vec![x + vec3(0.4, 0.0, 0.0), x - vec3(0.4, 0.0, 0.0)])
            .flat_map(|x| vec![x + vec3(0.0, 0.0, 0.4), x - vec3(0.0, 0.0, 0.4)])
            .map(|x| self.get_block(x))
            .all(|x| x == 0)
    }

    pub fn run_cmd_list(
        &mut self,
        l: (Vec<(Chunk, (u32, u32, u32))>, Vec<Vec<Vec<(u8, u8, u8)>>>),
        origin: IVec3,
    ) {
        self.origin = origin;
        for i in l.0 {
            if let Some(ref c) = i.0 {
                self.pix_buf.write(
                    &c.iter()
                        .flat_map(|x| x.iter())
                        .flat_map(|x| x.iter())
                        .cloned()
                        .collect::<Vec<Block>>(),
                );
                let s = CHUNK_SIZE as u32;
                let i = i.1;
                let i = (i.0 * s, i.1 * s, i.2 * s);
                self.block_buf.main_level().raw_upload_from_pixel_buffer(
                    self.pix_buf.as_slice(),
                    i.0..i.0 + s,
                    i.1..i.1 + s,
                    i.2..i.2 + s,
                );
            }
            self.chunks
                .insert(((i.1).2 as u8, (i.1).1 as u8, (i.1).0 as u8), i.0);
        }
        self.dpix_buf.write(
            &l.1.iter()
                .flat_map(|x| x.iter())
                .flat_map(|x| x.iter())
                .map(|x| {
                    if self.chunks[x].is_some() {
                        *x
                    } else {
                        (255, 255, 255)
                    }
                })
                // .cloned()
                .collect::<Vec<(u8, u8, u8)>>(),
        );
        self.chunk_buf.main_level().raw_upload_from_pixel_buffer(
            self.dpix_buf.as_slice(),
            0..CHUNK_NUM as u32,
            0..CHUNK_NUM as u32,
            0..CHUNK_NUM as u32,
        );
        self.map = l.1;
    }

    pub fn game_loop(&mut self) {
        use enum_iterator::IntoEnumIterator;

        let mut events_loop = None;
        std::mem::swap(&mut self.events_loop, &mut events_loop);
        let mut events_loop = events_loop.unwrap();

        let vertexes = vec![vert(-3.0, -3.0), vert(3.0, -3.0), vert(0.0, 3.0)];
        let vbuff = glium::VertexBuffer::new(&self.display, &vertexes).unwrap();
        let indices = glium::index::NoIndices(glium::index::PrimitiveType::TriangleStrip);

        let vshader = shader("vert.glsl".to_string(), &[]);
        let fshader = shader(
            "frag.glsl".to_string(),
            &["sky.glsl".to_string(), "bsdf.glsl".to_string()],
        );

        let program = glium::Program::from_source(&self.display, &vshader, &fshader, None).unwrap();

        let initial_time = 0.0; /*
                                6.0 // 06:00, in minutes
                                * 60.0 // Seconds
                                ;*/

        let mats = Material::into_enum_iter()
            .map(|x| x.mat_data())
            .collect::<Vec<MatData>>();
        let mat_buf = glium::uniforms::UniformBuffer::empty_unsized_immutable(
            &self.display,
            std::mem::size_of::<MatData>() * mats.len(),
        )
        .unwrap();
        mat_buf.write(mats.as_slice());

        let mut closed = false;
        let timer = stopwatch::Stopwatch::start_new();
        let mut last = timer.elapsed_ms();
        let camera_dir = vec3(0.0, 0.0, 1.0);
        let (mut rx, mut ry) = (0.0, 0.0);

        while !closed {
            let cur = timer.elapsed_ms();
            let delta = cur - last;
            // println!("FPS: {}", 1000 / delta.max(1));
            last = cur;

            let mut target = self.display.draw();
            target.clear_color(0.0, 0.0, 1.0, 1.0);

            let res = target.get_dimensions();
            let res = vec2(res.0 as f32, res.1 as f32);

            let camera_up = vec3(0.0, 1.0, 0.0);
            let q = glm::ext::rotate(&Matrix4::one(), rx / res.x * -6.28, camera_up)
                * camera_dir.extend(1.0);
            let camera_dir = normalize(vec3(q.x, q.y, q.z));
            let right = cross(camera_dir, camera_up);
            let q = glm::ext::rotate(&Matrix4::one(), ry / res.y * -6.28, right) * q;
            let camera_dir = normalize(vec3(q.x, q.y, q.z));

            events_loop.poll_events(|event| match event {
                glutin::Event::WindowEvent { event, .. } => match event {
                    glutin::WindowEvent::CloseRequested => closed = true,
                    glutin::WindowEvent::KeyboardInput { input, .. } => {
                        if let glutin::ElementState::Released = input.state {
                            match input.virtual_keycode {
                                Some(FORWARD) | Some(BACK) => {
                                    self.state.fb = 0.0;
                                }
                                Some(glutin::VirtualKeyCode::Space)
                                | Some(glutin::VirtualKeyCode::LShift) => {
                                    self.state.ud = 0.0;
                                }
                                Some(LEFT) | Some(RIGHT) => {
                                    self.state.lr = 0.0;
                                }
                                Some(glutin::VirtualKeyCode::LControl) => {
                                    self.state.m = 1.0;
                                }
                                Some(FLY) => {
                                    self.state.fly = !self.state.fly;
                                }
                                _ => (),
                            }
                        }
                        if let glutin::ElementState::Pressed = input.state {
                            match input.virtual_keycode {
                                Some(FORWARD) => {
                                    self.state.fb = 1.0;
                                }
                                Some(BACK) => {
                                    self.state.fb = -1.0;
                                }
                                Some(glutin::VirtualKeyCode::Space) => {
                                    self.state.ud = 1.0;
                                    self.state.try_jump = true;
                                }
                                Some(glutin::VirtualKeyCode::LShift) => {
                                    self.state.ud = -1.0;
                                }
                                Some(RIGHT) => {
                                    self.state.lr = 1.0;
                                }
                                Some(LEFT) => {
                                    self.state.lr = -1.0;
                                }
                                Some(glutin::VirtualKeyCode::LControl) => {
                                    self.state.m = 4.0;
                                }
                                _ => (),
                            }
                        }
                    }
                    _ => (),
                },
                glutin::Event::DeviceEvent { event, .. } => match event {
                    glutin::DeviceEvent::MouseMotion { delta } => {
                        rx += delta.0 as f32;
                        ry += delta.1 as f32;
                        ry = clamp(ry, -res.y * 0.25, res.y * 0.25);
                    }
                    // Button 1 is left-click
                    glutin::DeviceEvent::Button {
                        button: 1,
                        state: glutin::ElementState::Pressed,
                    } => {
                        if let Some(p) = self.trace(self.pos, camera_dir, 32) {
                            // println!("Setting block {:?}", p);
                            // self.set_block(to_ivec3(self.pos) - self.origin * CHUNK_SIZE as i32 + CHUNK_SIZE as i32 * CHUNK_NUM as i32 / 2, 0);
                            self.set_block(p, 0);
                        }
                    }
                    // Button 3 is right-click
                    glutin::DeviceEvent::Button {
                        button: 3,
                        state: glutin::ElementState::Pressed,
                    } => {
                        if let Some((p, n)) = self.trace_n(self.pos, camera_dir, 32) {
                            let p = p + n;
                            // println!("Setting block {:?}", p);
                            // self.set_block(to_ivec3(self.pos) - self.origin * CHUNK_SIZE as i32 + CHUNK_SIZE as i32 * CHUNK_NUM as i32 / 2, 0);
                            self.set_block(p, Material::Grass as u16);
                        }
                    }
                    _ => (),
                },
                _ => (),
            });

            self.do_move(
                vec3(1.0, 0.0, 1.0) * camera_dir,
                self.state.fb * self.state.m * (delta as f64 * MOVE_SPEED as f64) as f32,
            );
            self.do_move(
                vec3(1.0, 0.0, 1.0) * right,
                self.state.lr * self.state.m * (delta as f64 * MOVE_SPEED as f64) as f32,
            );

            // Jumping
            if self.state.fly {
                self.do_move(
                    vec3(0.0, 1.0, 0.0),
                    self.state.ud * self.state.m * (delta as f64 * MOVE_SPEED as f64) as f32,
                );
            } else {
                // We're not in the air
                let block_below = !self.can_move(self.pos - vec3(0.0, 0.2, 0.0));
                if block_below {
                    if self.state.jump > 0.3 {
                        self.state.try_jump = false;
                    }
                    self.state.jump = 0.0;
                }
                let g_force = 9.7; // m/s^2
                let j_seconds = self.state.jump; // s
                let total_g = j_seconds * g_force; // m/s
                let initial_y_v = if self.state.try_jump { 5.0 } else { 0.0 }; // m/s; this number was tuned manually, and only provisionally
                let current_v = initial_y_v - total_g; // m/s
                let current_m = current_v * (delta as f64 * 0.001 as f64) as f32; // meters

                self.do_move(vec3(0.0,1.0,0.0), current_m);

                if !block_below {
                    self.state.jump += (delta as f64 * 0.001 as f64) as f32;
                }
            }

            self.channel.0.send(Message::Move(self.pos)).unwrap();
            if let Ok(cmd) = self.channel.1.try_recv() {
                match cmd {
                    Message::ChunkMove(a, b, c) => {
                        self.run_cmd_list((a, b), c);
                    }
                    _ => (),
                }
            }

            target
                .draw(
                    &vbuff,
                    &indices,
                    &program,
                    &uniform! {
                        iTime: initial_time + timer.elapsed_ms() as f32 / 1000.0,
                        iResolution: *res.as_array(),
                        cameraPos: *self.pos.as_array(),
                        cameraDir: *camera_dir.as_array(),
                        cameraUp: *camera_up.as_array(),
                        chunks: &self.chunk_buf,
                        blocks: &self.block_buf,
                        chunk_origin: self.origin_u(),
                        mat_list: &mat_buf,
                    },
                    &Default::default(),
                )
                .unwrap();

            target.finish().unwrap();
        }
        self.channel.0.send(Message::Leave).unwrap();
    }

    pub fn origin_u(&self) -> [f32; 3] {
        *to_vec3(self.origin * CHUNK_SIZE as i32).as_array()
    }
}
