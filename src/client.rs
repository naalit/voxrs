use std::sync::Arc;
use crate::common::*;
use crate::input::*;
use crate::mesh::*;
use enum_iterator::IntoEnumIterator;
use crate::num_traits::One;
use glium::glutin::*;
use glium::*;
use glsl_include::Context as ShaderContext;
use rayon::prelude::*;
use std::collections::HashMap;
use crate::client_aux::*;
use std::sync::mpsc::*;

struct DrawStuff<'a> {
    params: DrawParameters<'a>,
    program: Program,
    mat_buf: glium::uniforms::UniformBuffer<[MatData]>,
}

#[derive(Clone)]
struct Camera {
    pos: Vec3,
    dir: Vec3,
    rx: f32,
    ry: f32,
    moving: Vec3, // x is forward, y is up, z is right
}

impl Camera {
    pub fn new(pos: Vec3) -> Camera {
        Camera {
            pos,
            dir: vec3(0.0, 0.0, 1.0),
            rx: 0.0,
            ry: 0.0,
            moving: vec3(0.0, 0.0, 0.0),
        }
    }

    pub fn event(&mut self, event: DeviceEvent, resolution: (u32, u32)) {
        match event {
            glutin::DeviceEvent::MouseMotion { delta } => {
                self.rx += delta.0 as f32;
                self.ry += delta.1 as f32;
                self.ry = glm::clamp(
                    self.ry,
                    -(resolution.1 as f32 * 0.25),
                    resolution.1 as f32 * 0.25,
                );
            }
            glutin::DeviceEvent::Key(glutin::KeyboardInput {
                scancode,
                state: glutin::ElementState::Pressed,
                ..
            }) => {
                match num_traits::FromPrimitive::from_u32(scancode).unwrap_or(KeyPress::Nothing) {
                    KeyPress::Forward => self.moving.x = 1.0,
                    KeyPress::Back => self.moving.x = -1.0,
                    _ => (),
                }
            }
            glutin::DeviceEvent::Key(glutin::KeyboardInput {
                scancode,
                state: glutin::ElementState::Released,
                ..
            }) => {
                match num_traits::FromPrimitive::from_u32(scancode).unwrap_or(KeyPress::Nothing) {
                    KeyPress::Forward | KeyPress::Back => self.moving.x = 0.0,
                    _ => (),
                }
            }
            _ => (),
        };
    }

    pub fn update(&mut self, delta: f64, resolution: (u32, u32)) {
        self.pos = self.pos + self.dir * self.moving.x * delta as f32 * 8.0;

        let camera_up = vec3(0.0, 1.0, 0.0);
        let q = glm::ext::rotate(
            &Matrix4::one(),
            self.rx / resolution.0 as f32 * -6.28,
            camera_up,
        ) * vec4(0.0, 0.0, 1.0, 1.0);
        self.dir = normalize(vec3(q.x, q.y, q.z));
        let camera_right = cross(self.dir, camera_up);
        let q = glm::ext::rotate(
            &Matrix4::one(),
            self.ry / resolution.1 as f32 * -6.28,
            camera_right,
        ) * q;
        self.dir = normalize(vec3(q.x, q.y, q.z));
    }

    pub fn mat(&self, resolution: (u32, u32)) -> [[f32; 4]; 4] {
        let camera_up = vec3(0.0, 1.0, 0.0);
        let camera_right = cross(self.dir, camera_up);
        let camera_up = cross(camera_right, self.dir);

        let proj_mat = glm::ext::perspective(
            radians(90.0),
            resolution.0 as f32 / resolution.1 as f32,
            0.1,
            200.0,
        );
        let proj_mat = proj_mat * glm::ext::look_at_rh(self.pos, self.pos + self.dir, camera_up);
        let proj_mat_arr = [
            *proj_mat[0].as_array(),
            *proj_mat[1].as_array(),
            *proj_mat[2].as_array(),
            *proj_mat[3].as_array(),
        ];

        proj_mat_arr
    }
}

/// Load a shader, replacing any ``#include` declarations to files in `includes`
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

pub struct Client<'a> {
    camera: Camera,
    chunks: HashMap<(i32, i32, i32), Arc<Chunk>>,
    meshes: HashMap<(i32, i32, i32), Mesh>,
    evloop: EventsLoop,
    display: Display,
    draw_stuff: DrawStuff<'a>,
    aux: (Sender<Message>, Receiver<ClientMessage>),
}

impl<'a> Client<'a> {
    pub fn new(display: Display, evloop: EventsLoop, conn: Connection, player: Vec3) -> Self {
        let (to, from_them) = channel();
        let (to_them, from) = channel();
        std::thread::spawn(move || {
            client_aux_thread(conn, (to_them, from_them), player)
        });

        let vshader = shader("vert.glsl".to_string(), &[]);
        let fshader = shader("frag.glsl".to_string(), &[]);
        let program = glium::Program::from_source(&display, &vshader, &fshader, None).unwrap();

        let mats = Material::into_enum_iter()
            .map(|x| x.mat_data())
            .collect::<Vec<MatData>>();
        let mat_buf = glium::uniforms::UniformBuffer::empty_unsized_immutable(&display, std::mem::size_of::<MatData>() * mats.len()).unwrap();
        mat_buf.write(mats.as_slice());

        Client {
            camera: Camera::new(player),
            chunks: HashMap::with_capacity((CHUNK_NUM.x*CHUNK_NUM.y*CHUNK_NUM.z/2) as usize),
            meshes: HashMap::with_capacity((CHUNK_NUM.x*CHUNK_NUM.y*CHUNK_NUM.z/2) as usize),
            evloop,
            display,
            draw_stuff: DrawStuff {
                params: glium::DrawParameters {
                    depth: glium::Depth {
                        test: glium::draw_parameters::DepthTest::IfLess,
                        write: true,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                program,
                mat_buf,
            },
            aux: (to, from),
        }
    }

    /// The player position
    pub fn pos(&self) -> Vec3 {
        self.camera.pos
    }

    pub fn display(&self) -> &Display {
        &self.display
    }

    pub fn draw(&mut self, target: &mut Frame) {
        target.clear_color(0.0, 0.0, 0.0, 1.0);
        target.clear_depth(1.0);

        let resolution = target.get_dimensions();

        let proj_mat = self.camera.mat(resolution);

        for (_loc, mesh) in self.meshes.iter() {
            mesh.draw(
                target,
                &self.draw_stuff.program,
                &self.draw_stuff.params,
                uniform! {
                    proj_mat: proj_mat,
                    mat_buf: &self.draw_stuff.mat_buf,
                    camera_pos: *self.camera.pos.as_array(),
                },
            );
        }
    }

    pub fn update(&mut self, delta: f64) -> bool {
        let mut open = true;
        let resolution: (u32, u32) = self
            .display
            .gl_window()
            .window()
            .get_inner_size()
            .unwrap()
            .into();

        let mut camera = self.camera.clone(); // Because we can't borrow self.camera in the closure
        self.evloop.poll_events(|event| match event {
            glutin::Event::WindowEvent {
                event: glutin::WindowEvent::CloseRequested,
                ..
            } => open = false,
            glutin::Event::DeviceEvent { event, .. } => {
                match event {
                    _ => (),
                }
                camera.event(event, resolution);
            }
            _ => (),
        });
        camera.update(delta, resolution);
        self.camera = camera;

        if let Ok(chunks) = self.aux.1.try_recv() {
            // Only load chunks once per frame
            self.load_chunks(chunks);
        }
        self.aux.0.send(Message::PlayerMove(self.camera.pos)).unwrap();

        open
    }

    /// Load a bunch of chunks at once. Prunes the root as well
    /// Uploads everything to the GPU
    pub fn load_chunks(&mut self, chunks: Vec<(IVec3, Vec<crate::mesh::Vertex>, Arc<Chunk>)>) {
        self.prune_chunks();

        for (i, v, c) in chunks {
            let mesh = Mesh::new(
                &self.display,
                v,
                to_vec3(i) * CHUNK_SIZE,
                vec3(0.0, 0.0, 0.0),
            );

            self.chunks.insert(as_tuple(i), c);
            self.meshes.insert(as_tuple(i), mesh);
        }
    }

    /// Unload the chunk at position `idx` in world space.
    /// This is the client function, so it won't store it anywhere or anything, that's the server's job.
    pub fn unload(&mut self, idx: IVec3) {
        self.chunks.remove(&as_tuple(idx));
        self.meshes.remove(&as_tuple(idx));
    }

    /// Unloads chunks that are too far away
    fn prune_chunks(&mut self) {
        for i in self.chunks.clone().keys() {
            let i = as_vec(*i);
            let p = chunk_to_world(i);
            let d = length(p - self.camera.pos);
            if d > DRAW_DIST {
                self.unload(i);
            }
        }
    }
}
