use crate::client_aux::*;
use crate::common::*;
use crate::input::*;
use crate::mesh::*;
use crate::num_traits::One;
use enum_iterator::IntoEnumIterator;
use glium::glutin::*;
use glium::*;
use glsl_include::Context as ShaderContext;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::mpsc::*;
use std::sync::{Arc, RwLock};

#[derive(Clone)]
struct Camera {
    pos: Point3<f32>,
    dir: Vec3,
    rx: f32,
    ry: f32,
    moving: Vec3, // x is forward, y is up, z is right
}

impl Camera {
    pub fn new(pos: Point3<f32>) -> Camera {
        Camera {
            pos,
            dir: Vec3::new(0.0, 0.0, 1.0),
            rx: 0.0,
            ry: 0.0,
            moving: Vec3::new(0.0, 0.0, 0.0),
        }
    }

    pub fn event(&mut self, event: DeviceEvent, resolution: (u32, u32)) {
        match event {
            glutin::DeviceEvent::MouseMotion { delta } => {
                self.rx += delta.0 as f32;
                self.ry += delta.1 as f32;
                self.ry = na::clamp(
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

    pub fn update(&mut self, delta: f64, resolution: (u32, u32), player: &mut np::object::Body<f32>) {
        let player = player.downcast_mut::<np::object::RigidBody<f32>>().unwrap();
        player.apply_force(0, &np::algebra::Force3::new(self.dir * self.moving.x * 50.0, Vec3::zeros()), np::algebra::ForceType::Impulse, true);
        self.pos = Point3::from(player.position().translation.vector);
//        self.pos = self.pos + self.dir * self.moving.x * delta as f32 * 8.0;

        let camera_up = Unit::new_normalize(Vec3::new(0.0, 1.0, 0.0));
        let q = na::Rotation3::from_axis_angle(&camera_up, self.rx / resolution.0 as f32 * -6.28)
            * Vec3::new(0.0, 0.0, 1.0);
        self.dir = q.normalize();
        let camera_right = Unit::new_normalize(self.dir.cross(&camera_up));
        let q =
            na::Rotation3::from_axis_angle(&camera_right, self.ry / resolution.1 as f32 * -6.28)
                * q;
        self.dir = q.normalize();
    }

    pub fn mat(&self, resolution: (u32, u32)) -> [[f32; 4]; 4] {
        let camera_up = Vec3::new(0.0, 1.0, 0.0);
        let camera_right = Unit::new_normalize(self.dir.cross(&camera_up));
        let camera_up = camera_right.cross(&self.dir);

        let proj_mat = na::Matrix4::new_perspective(
            resolution.0 as f32 / resolution.1 as f32,
            radians(90.0),
            0.1,
            200.0,
        );
        let proj_mat =
            proj_mat * na::Matrix4::look_at_rh(&self.pos, &(self.pos + self.dir), &camera_up);
        proj_mat.into()
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

#[derive(Clone, Copy)]
struct SimpleVert {
    p: [f32; 2],
}
implement_vertex!(SimpleVert, p);

struct DrawStuff<'a> {
    params: DrawParameters<'a>,
    gbuff_program: Program,
    shade_program: Program,
    gbuff: glium::texture::Texture2d,
    gbuff_depth: glium::texture::depth_texture2d::DepthTexture2d,
    quad: VertexBuffer<SimpleVert>,
    mat_buf: glium::uniforms::UniformBuffer<[MatData]>,
}

impl<'a> DrawStuff<'a> {
    fn new(display: &Display, resolution: (u32, u32), config: Arc<ClientConfig>) -> Self {
        let vshader = shader("gbuff.vert".to_string(), &[]);
        let fshader = shader("gbuff.frag".to_string(), &[]);
        let gbuff_program = glium::Program::from_source(display, &vshader, &fshader, None).unwrap();

        let vshader = shader("blank.vert".to_string(), &[]);
        let fshader = shader("shade.frag".to_string(), &[]);
        let shade_program = glium::Program::from_source(display, &vshader, &fshader, None).unwrap();

        let quad = [
            SimpleVert { p: [-1.0, -1.0] },
            SimpleVert { p: [-1.0, 1.0] },
            SimpleVert { p: [1.0, -1.0] },
            SimpleVert { p: [1.0, 1.0] },
        ];
        let quad = glium::VertexBuffer::new(display, &quad).unwrap();

        let mats = Material::into_enum_iter()
            .map(|x| x.mat_data())
            .collect::<Vec<MatData>>();
        let mat_buf = glium::uniforms::UniformBuffer::empty_unsized_immutable(
            display,
            std::mem::size_of::<MatData>() * mats.len(),
        )
        .unwrap();
        mat_buf.write(mats.as_slice());

        let gbuff = glium::texture::Texture2d::empty_with_format(
            display,
            glium::texture::UncompressedFloatFormat::F32F32F32F32,
            glium::texture::MipmapsOption::NoMipmap,
            resolution.0,
            resolution.1,
        )
        .unwrap();
        let gbuff_depth = glium::texture::depth_texture2d::DepthTexture2d::empty(
            display,
            resolution.0,
            resolution.1,
        )
        .unwrap();
        DrawStuff {
            params: glium::DrawParameters {
                depth: glium::Depth {
                    test: glium::draw_parameters::DepthTest::IfLess,
                    write: true,
                    ..Default::default()
                },
                polygon_mode: if config.wireframe { glium::draw_parameters::PolygonMode::Line } else { glium::draw_parameters::PolygonMode::Fill },
                line_width: if config.wireframe { Some(2.0) } else { None },
                ..Default::default()
            },
            gbuff_program,
            shade_program,
            gbuff,
            gbuff_depth,
            quad,
            mat_buf,
        }
    }
}

pub struct Client {
    camera: Camera,
    chunks: HashMap<IVec3, Arc<RwLock<Chunk>>>,
    meshes: HashMap<IVec3, Mesh>,
    colliders: HashMap<IVec3, np::object::ColliderHandle>,
    display: Display,
    aux: (Sender<Message>, Receiver<ClientMessage>),
    world: np::world::World<f32>,
    player_handle: np::object::BodyHandle,
    config: Arc<ClientConfig>,
}

impl Client {
    pub fn new(display: Display, conn: Connection, player: Vec3) -> Self {
        let config = Arc::new(ClientConfig {
            mesher: Box::new(Greedy),
            wireframe: false,
            game_config: GameConfig {},
        });

        let (to, from_them) = channel();
        let (to_them, from) = channel();
        let two = Arc::clone(&config);
        std::thread::spawn(move || client_aux_thread(conn, (to_them, from_them), player, two));
        let mut world = np::world::World::new();
        world.set_gravity(Vec3::y() * -9.81);

        let player_shape = nc::shape::ShapeHandle::new(nc::shape::Capsule::new(1.0, 0.25));
        let player_collider = np::object::ColliderDesc::new(player_shape);
        let player_handle = np::object::RigidBodyDesc::new()
            .collider(&player_collider)
            .translation(player)
            .mass(90.0) // In kg, and this person is 2 meters tall
            .build(&mut world)
            .handle();

        Client {
            camera: Camera::new(player.into()),
            chunks: HashMap::with_capacity((CHUNK_NUM.0 * CHUNK_NUM.1 * CHUNK_NUM.2 / 2) as usize),
            meshes: HashMap::with_capacity((CHUNK_NUM.0 * CHUNK_NUM.1 * CHUNK_NUM.2 / 2) as usize),
            colliders: HashMap::with_capacity((CHUNK_NUM.0 * CHUNK_NUM.1 * CHUNK_NUM.2 / 2) as usize),
            display,
            aux: (to, from),
            world,
            player_handle,
            config,
        }
    }

    /// The player position
    pub fn pos(&self) -> Vec3 {
        self.camera.pos.coords
    }

    pub fn display(&self) -> &Display {
        &self.display
    }

    fn draw(
        &self,
        target: &mut Frame,
        draw_stuff: &DrawStuff,
        gbuff_fb: &mut glium::framebuffer::SimpleFrameBuffer,
    ) {
        target.clear_color(0.0, 0.0, 0.0, 1.0);
        target.clear_depth(1.0);

        let resolution = target.get_dimensions();

        let proj_mat: [[f32; 4]; 4] = self.camera.mat(resolution);

        // Draw chunks onto the G-Buffer
        gbuff_fb.clear_color_and_depth((0.0, 0.0, 0.0, 0.0), 1.0);
        for (_loc, mesh) in self.meshes.iter() {
            mesh.draw(
                gbuff_fb,
                &draw_stuff.gbuff_program,
                &draw_stuff.params,
                uniform! {
                    proj_mat: proj_mat,
                    // mat_buf: &self.draw_stuff.mat_buf,
                    // camera_pos: *self.camera.pos.as_array(),
                },
            );
        }

        // Draw a fullscreen quad and shade using the G-Buffer
        target
            .draw(
                &draw_stuff.quad,
                &glium::index::NoIndices(glium::index::PrimitiveType::TriangleStrip),
                &draw_stuff.shade_program,
                &uniform! {
                    mat_buf: &draw_stuff.mat_buf,
                    camera_pos: <[f32; 3]>::from(self.pos().into()),
                    gbuff: &draw_stuff.gbuff,
                },
                &Default::default(),
            )
            .unwrap();
    }

    pub fn update(&mut self, evloop: &mut EventsLoop, delta: f64) -> bool {
        let mut open = true;
        let resolution: (u32, u32) = self
            .display
            .gl_window()
            .window()
            .get_inner_size()
            .unwrap()
            .into();

        if self.meshes.len() != 0 {
            self.world.step(); // TODO: make timestep match delta?
        }

        let mut camera = self.camera.clone(); // Because we can't borrow self.camera in the closure
        evloop.poll_events(|event| match event {
            glutin::Event::WindowEvent {
                event: glutin::WindowEvent::CloseRequested,
                ..
            } => open = false,
            glutin::Event::DeviceEvent { event, .. } => {
                match event {
                    // Left-click
                    glutin::DeviceEvent::Button {
                        button: 1,
                        state: glutin::ElementState::Pressed,
                    } => {
                        if let Some(p) = self.trace(self.pos(), self.camera.dir, 32) {
                            self.set_block(p.cell, Material::Air);
                        }
                    }
                    _ => (),
                }
                camera.event(event, resolution);
            }
            _ => (),
        });
        camera.update(delta, resolution, self.world.body_mut(self.player_handle).unwrap());
        self.camera = camera;

        if let Ok(chunks) = self.aux.1.try_recv() {
            // Only load chunks once per frame
            self.load_chunks(chunks);
        }
        self.aux.0.send(Message::PlayerMove(self.pos())).unwrap();

        open
    }

    /// Runs the game loop on the client side
    pub fn game_loop(mut self, resolution: (u32, u32), mut evloop: EventsLoop) {
        let draw_stuff = DrawStuff::new(&self.display, resolution, Arc::clone(&self.config));

        let mut gbuff_fb = glium::framebuffer::SimpleFrameBuffer::with_depth_buffer(
            &self.display,
            &draw_stuff.gbuff,
            &draw_stuff.gbuff_depth,
        )
        .unwrap();

        let mut timer = stopwatch::Stopwatch::start_new();

        let mut open = true;
        while open {
            let delta = timer.elapsed_ms() as f64 / 1000.0;
            println!("{:.1} FPS", 1.0 / delta);
            timer.restart();

            let mut target = self.display().draw();

            self.draw(&mut target, &draw_stuff, &mut gbuff_fb);

            // Most computation should go after this point, while the GPU is rendering

            open = self.update(&mut evloop, delta);

            target.finish().unwrap();
        }
    }

    pub fn set_block(&mut self, loc: IVec3, new: Material) {
        println!("Setting {:?} to {:?}", loc, new);
        let chunk = loc / CHUNK_SIZE as i32;
        let in_chunk = na::wrap(loc, Vector3::zeros(), Vector3::repeat(CHUNK_SIZE as i32)); //((loc % CHUNK_SIZE as i32) + CHUNK_SIZE as i32) % CHUNK_SIZE as i32;

        let chunk_rc = self.chunks.get(&chunk).unwrap();
        chunk_rc.write().unwrap()[in_chunk.x as usize][in_chunk.y as usize][in_chunk.z as usize] =
            new;
        let verts = self.config.mesher.mesh(&chunk_rc.read().unwrap());
        let mesh = Mesh::new(
            &self.display,
            verts,
            chunk.map(|x| x as f32) * CHUNK_SIZE,
            Vec3::new(0.0, 0.0, 0.0),
        );
        self.meshes.insert(chunk, mesh); //.expect(&format!("Chunk {:?}", chunk));
    }

    /// Load a bunch of chunks at once. Prunes the root as well
    /// Uploads everything to the GPU
    pub fn load_chunks(
        &mut self,
        chunks: ClientMessage,
    ) {
        self.prune_chunks();

        for (i, v, s, c) in chunks {
            // TODO indices
            if let Some(chunk_shape) = s {
                let chunk_collider = np::object::ColliderDesc::new(chunk_shape)
                    .translation(i.map(|x| x as f32) * CHUNK_SIZE)
                    .build_with_parent(np::object::BodyPartHandle::ground(), &mut self.world)
                    .unwrap();
                self.colliders.insert(i, chunk_collider.handle());
            }

            let mesh = Mesh::new(
                &self.display,
                v,
                i.map(|x| x as f32) * CHUNK_SIZE,
                Vec3::new(0.0, 0.0, 0.0),
            );

            self.chunks.insert(i, c);
            self.meshes.insert(i, mesh);
        }
    }

    /// Unload the chunk at position `idx` in world space.
    /// This is the client function, so it won't store it anywhere or anything, that's the server's job.
    pub fn unload(&mut self, idx: IVec3) {
        self.chunks.remove(&idx);
        self.meshes.remove(&idx);
        if let Some(handle) = self.colliders.remove(&idx) {
            self.world.remove_colliders(&[handle]);
        }
    }

    /// Unloads chunks that are too far away
    fn prune_chunks(&mut self) {
        for &i in self.chunks.clone().keys() {
            let p = chunk_to_world(i);
            let d = (p - self.pos()).norm();
            if d > DRAW_DIST {
                self.unload(i);
            }
        }
    }
}

pub struct Intersection {
    pub pos: Vec3,
    pub cell: IVec3,
    pub t: f32,
    pub normal: Vec3,
}
/// Computes the analytic intersection of a ray with an axis-aligned cube with side length 1
/// We actually only compute tmin, and we assume that it is a valid intersection (since this is only called by voxel traversal)
fn isect_cube(cell: IVec3, ro: Vec3, rd: Vec3) -> Intersection {
    let mn = cell.map(|x| x as f32) - Vec3::repeat(0.5);
    let mx = mn + Vec3::repeat(1.0);
    let t1 = (mn - ro).component_div(&rd);
    let t2 = (mx - ro).component_div(&rd);
    let t = t1.zip_map(&t2, |x, y| x.min(y)).max(); // tmin
    let pos = ro + rd * t;
    let normal = pos - cell.map(|x| x as f32);
    let m = normal.max();
    let normal = normal.map(|x| if x == m { 1.0 } else { 0.0 });
    Intersection {
        pos,
        cell,
        t,
        normal,
    }
}

impl Client {
    pub fn trace(&self, ro: Vec3, rd: Vec3, max_iter: u32) -> Option<Intersection> {
        let ro_chunk = world_to_chunk(ro);

        let mut ipos = ro.map(|x| x as i32);
        let mut last_chunk_p = ipos / CHUNK_SIZE as i32;
        assert_eq!(ro_chunk, last_chunk_p);
        let mut last_chunk_rc = self.chunks.get(&last_chunk_p)?;
        let tdelta = rd.map(|x| 1.0 / x.abs());
        let istep = rd.map(|x| x.signum() as i32);
        let srd = rd.map(|x| x.signum());
        let mut side_dist =
            (srd.component_mul(&(ro.map(|x| x.floor()) - ro)) + (srd * 0.5) + Vec3::repeat(0.5))
                .component_mul(&tdelta);

        for _ in 0..max_iter {
            let cur_chunk = ipos / CHUNK_SIZE as i32;
            if cur_chunk != last_chunk_p {
                last_chunk_p = cur_chunk;
                last_chunk_rc = self.chunks.get(&last_chunk_p)?;
            }
            // This makes sure to wrap around negatives and get the right numbers
            let in_chunk = na::wrap(ipos, Vector3::zeros(), Vector3::repeat(CHUNK_SIZE as i32)); //((ipos % CHUNK_SIZE as i32) + CHUNK_SIZE as i32) % CHUNK_SIZE as i32;

            let block = last_chunk_rc.read().unwrap()[in_chunk.x as usize][in_chunk.y as usize]
                [in_chunk.z as usize];
            if !block.pick_through() {
                return Some(isect_cube(ipos, ro, rd));
            }

            // Advance
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
}
