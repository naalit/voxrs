use super::chunk::*;
use super::common::*;
use super::material::*;
use num_traits::identities::One;
use glium::backend::Facade;
use glium::*;
use glsl_include::Context as ShaderContext;
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
    channel: (Sender<Vec3>, Receiver<CommandList>),
}

impl Client {
    pub fn new(
        chunks: &ChunksU,
        chunks_new: HashMap<(u8, u8, u8), Chunk>,
        pos: Vec3,
        channel: (Sender<Vec3>, Receiver<CommandList>),
    ) -> Self {
        let events_loop = if cfg!(target_os = "linux") { glutin::os::unix::EventsLoopExt::new_x11().unwrap() } else { glutin::EventsLoop::new() };
        let wb = glutin::WindowBuilder::new()
            .with_title("Vox.rs")
            .with_fullscreen(Some(events_loop.get_primary_monitor()))
            .with_decorations(false);
        let cb = glutin::ContextBuilder::new(); //.with_vsync(true);
        let display = glium::Display::new(wb, cb, &events_loop).unwrap();
        display
            .gl_window()
            .window()
            .grab_cursor(true).unwrap();
        display
            .gl_window()
            .window()
            .hide_cursor(true);//.unwrap();
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

    pub fn get_block(&self, pos: Vec3) -> u16 {
        let chunk = chunk(pos) - self.origin + CHUNK_NUM as i32 / 2;
        let in_chunk = in_chunk(pos);
        let offset = self.map[chunk.z as usize][chunk.y as usize][chunk.x as usize];
        if offset.0 == 255 {
            0
        } else {
            self.chunks[&offset].map_or(0, |x| {
                x[in_chunk.z as usize][in_chunk.y as usize][in_chunk.x as usize]
            })
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

    pub fn run_cmd_list(&mut self, l: CommandList) {
        self.origin = l.2;
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
        let mut camera_dir = vec3(0.0,0.0,1.0);

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
            let right = cross(camera_dir, camera_up);

            events_loop.poll_events(|event| match event {
                glutin::Event::WindowEvent { event, .. } => match event {
                    glutin::WindowEvent::CloseRequested => closed = true,
                    glutin::WindowEvent::KeyboardInput { input, .. } => {
                        if let glutin::ElementState::Released = input.state {
                            match input.virtual_keycode {
                                Some(FORWARD)
                                | Some(BACK) => {
                                    self.state.fb = 0.0;
                                }
                                Some(glutin::VirtualKeyCode::Space)
                                | Some(glutin::VirtualKeyCode::LShift) => {
                                    self.state.ud = 0.0;
                                }
                                Some(LEFT)
                                | Some(RIGHT) => {
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
                        let q = glm::ext::rotate(&Matrix4::one(), delta.0 as f32 / res.x * -6.28, camera_up) * camera_dir.extend(1.0);
                        let q = glm::ext::rotate(&Matrix4::one(), delta.1 as f32 / res.y * -6.28, right) * q;
                        camera_dir = vec3(q.x,q.y,q.z);
                },
                _ => (),
                },
                _ => (),
            });
            let new_pos = self.pos
                + (vec3(1.0, 0.0, 1.0) * camera_dir * self.state.fb)
                    * self.state.m
                    * (delta as f64 * MOVE_SPEED as f64) as f32;
            if self.can_move(new_pos) {
                self.pos = new_pos;
            }
            let new_pos = self.pos
                + (vec3(1.0, 0.0, 1.0) * right * self.state.lr)
                    * self.state.m
                    * (delta as f64 * MOVE_SPEED as f64) as f32;
            if self.can_move(new_pos) {
                self.pos = new_pos;
            }

            // Jumping
            if self.state.fly {
                let new_pos = self.pos
                    + (vec3(0.0, 1.0, 0.0) * self.state.ud)
                        * self.state.m
                        * (delta as f64 * MOVE_SPEED as f64) as f32;
                if self.can_move(new_pos) {
                    self.pos = new_pos;
                }
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
                let new_pos = self.pos + vec3(0.0, 1.0, 0.0) * current_m;
                if self.can_move(new_pos) {
                    self.pos = new_pos;
                }
                //  else if current_m > 1.0 - abs(fract(self.pos.y)) {
                //     // We can get up to insane speeds when falling from heights, so we allow this partial movement
                //     self.pos.y = ceil(self.pos.y) - 0.1;
                // } else if current_m < -abs(fract(self.pos.y)) {
                //     self.pos.y = floor(self.pos.y) + 0.1;
                // }
                if !block_below {
                    self.state.jump += (delta as f64 * 0.001 as f64) as f32;
                }
            }

            self.channel.0.send(self.pos).unwrap();
            if let Ok(cmd) = self.channel.1.try_recv() {
                // println!("Accepting server chunks");
                self.run_cmd_list(cmd);
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
    }

    pub fn origin_u(&self) -> [f32; 3] {
        *to_vec3(self.origin * CHUNK_SIZE as i32).as_array()
    }
}
