use crate::client_aux::*;
use crate::common::*;
use crate::input::*;
use crate::mesh::*;
use enum_iterator::IntoEnumIterator;
use glium::glutin::*;
use glium::*;
use glsl_include::Context as ShaderContext;
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

    pub fn update(
        &mut self,
        delta: f64,
        resolution: (u32, u32),
        player: &mut np::object::Body<f32>,
    ) {
        let player = player.downcast_mut::<np::object::RigidBody<f32>>().unwrap();
        player.apply_force(
            0,
            &np::algebra::Force3::new(self.dir * self.moving.x * 50.0, Vec3::zeros()),
            np::algebra::ForceType::Impulse,
            true,
        );
        self.pos = Point3::from(player.position().translation.vector) + Vec3::new(0.0, 0.4, 0.0);
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
            6000.0,
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
        let fshader = shader("shade.frag".to_string(), &["sky.glsl".to_string()]);
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
                polygon_mode: if config.wireframe {
                    glium::draw_parameters::PolygonMode::Line
                } else {
                    glium::draw_parameters::PolygonMode::Fill
                },
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
    time: f64,
    world: np::world::World<f32>,
    player_handle: np::object::BodyHandle,
    config: Arc<ClientConfig>,
}

impl Client {
    pub fn new(
        display: Display,
        game_config: Arc<GameConfig>,
        conn: Connection,
        player: Vec3,
    ) -> Self {
        let config = Arc::new(ClientConfig {
            mesher: Box::new(Greedy),
            wireframe: false,
            batch_size: 16,
            game_config,
        });

        let (to, from_them) = channel();
        let (to_them, from) = channel();
        let two = Arc::clone(&config);
        std::thread::spawn(move || client_aux_thread(conn, (to_them, from_them), player, two));
        let mut world = np::world::World::new();
        world.set_gravity(Vec3::y() * -9.81);

        let player_shape = nc::shape::ShapeHandle::new(nc::shape::Capsule::new(0.65, 0.25));
        let player_collider = np::object::ColliderDesc::new(player_shape)
            .name("player".to_string())
            .margin(0.05);
        let player_handle = np::object::RigidBodyDesc::new()
            .collider(&player_collider)
            .translation(player)
            .mass(90.0) // In kg, and this person is 2 meters tall
            .build(&mut world)
            .handle();

        Client {
            camera: Camera::new(player.into()),
            chunks: HashMap::with_capacity(config.game_config.draw_chunks.pow(3) / 2),
            meshes: HashMap::with_capacity(config.game_config.draw_chunks.pow(3) / 2),
            colliders: HashMap::with_capacity(config.game_config.draw_chunks.pow(3) / 2),
            display,
            aux: (to, from),
            time: 0.0,
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

        let sun_speed = 0.05;
        let sun_dir = Vec3::new((self.time * sun_speed).sin() as f32, (self.time * sun_speed).cos() as f32, 0.0).normalize();

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
                    proj_mat: proj_mat,
                    sun_dir: <[f32; 3]>::from(sun_dir.into()),
                    gbuff: &draw_stuff.gbuff,
                },
                &Default::default(),
            )
            .unwrap();

        // Draw p2 chunks onto the G-Buffer
        for (_loc, mesh) in self.meshes.iter() {
            mesh.draw_p2(
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

        // Draw a fullscreen quad and shade using the p2 G-Buffer
        target
            .draw(
                &draw_stuff.quad,
                &glium::index::NoIndices(glium::index::PrimitiveType::TriangleStrip),
                &draw_stuff.shade_program,
                &uniform! {
                    mat_buf: &draw_stuff.mat_buf,
                    camera_pos: <[f32; 3]>::from(self.pos().into()),
                    proj_mat: proj_mat,
                    sun_dir: <[f32; 3]>::from(sun_dir.into()),
                    gbuff: &draw_stuff.gbuff,
                },
                &glium::draw_parameters::DrawParameters {
                    blend: Blend::alpha_blending(),
                    ..Default::default()
                },
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
                        if let Some(p) = self.trace(self.pos(), self.camera.dir, 16.0) {
                            self.set_block(p.cell, Material::Air);
                        }
                    }
                    _ => (),
                }
                camera.event(event, resolution);
            }
            _ => (),
        });
        camera.update(
            delta,
            resolution,
            self.world.body_mut(self.player_handle).unwrap(),
        );
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
            self.time += delta;

            // println!("{:.1} FPS", 1.0 / delta);
            timer.restart();

            let mut target = self.display().draw();

            self.draw(&mut target, &draw_stuff, &mut gbuff_fb);

            // Most computation should go after this point, while the GPU is rendering

            open = self.update(&mut evloop, delta);

            target.finish().unwrap();
        }
    }

    pub fn set_block(&mut self, loc: IVec3, new: Material) {
        let chunk = world_to_chunk(loc.map(|x| x as f32));
        let in_chunk = in_chunk(loc.map(|x| x as f32));

        let chunk_rc = self.chunks.get(&chunk).unwrap();
        chunk_rc.write().unwrap().set_block(in_chunk, new);

        self.remesh(chunk);
        for d in 0..3 {
            if in_chunk[d] == 0 {
                let mut i = chunk;
                i[d] -= 1;
                self.remesh(i);
            }
            if in_chunk[d] == CHUNK_SIZE as usize - 1 {
                let mut i = chunk;
                i[d] += 1;
                self.remesh(i);
            }
        }
    }

    fn remesh(&mut self, chunk: IVec3) -> Option<()> {
        let chunk_rc = self.chunks.get(&chunk)?;
        let neighbors: Vec<Arc<RwLock<Chunk>>> = neighbors(chunk)
            .into_iter()
            .filter_map(|x| self.chunks.get(&x))
            .cloned()
            .collect();
        // This is kind of messy, but it works
        if neighbors.len() != crate::mesh::neighbors(chunk).len() {
            return None;
        }

        if let Some(c) = self.colliders.remove(&chunk) {
            self.world.remove_colliders(&[c]);
        }

        let verts = self
            .config
            .mesher
            .mesh(&chunk_rc.read().unwrap(), neighbors.clone(), false);
        let verts_p2 = self
            .config
            .mesher
            .mesh(&chunk_rc.read().unwrap(), neighbors, true);

        if verts.len() != 0 {
            let v_physics: Vec<_> = verts.iter().map(|x| na::Point3::from(x.pos)).collect();
            let i_physics: Vec<_> = (0..v_physics.len() / 3)
                .map(|x| na::Point3::new(x * 3, x * 3 + 1, x * 3 + 2))
                .collect();
            let chunk_shape =
                nc::shape::ShapeHandle::new(nc::shape::TriMesh::new(v_physics, i_physics, None));


            let chunk_collider = np::object::ColliderDesc::new(chunk_shape)
                .translation(chunk.map(|x| x as f32) * CHUNK_SIZE)
                .build_with_parent(np::object::BodyPartHandle::ground(), &mut self.world)
                .unwrap();
            self.colliders.insert(chunk, chunk_collider.handle());
        }

        let mesh = Mesh::new(
            &self.display,
            verts,
            verts_p2,
            chunk.map(|x| x as f32) * CHUNK_SIZE,
            Vec3::new(0.0, 0.0, 0.0),
        );
        self.meshes.insert(chunk, mesh);

        Some(())
    }

    /// Load a bunch of chunks at once. Prunes the root as well
    /// Uploads everything to the GPU
    pub fn load_chunks(&mut self, chunks: ClientMessage) {
        self.prune_chunks();

        for (i, v, v2, s, c) in chunks {
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
                v2,
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
            let p = world_to_chunk(self.pos());
            let d = (p - i).map(|x| x as f32).norm();
            if d > self.config.game_config.draw_chunks as f32 {
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

impl Client {
    pub fn trace(&self, ro: Vec3, rd: Vec3, max_t: f32) -> Option<Intersection> {
        let ray = nc::query::Ray::new(ro.into(), rd);

        let g = nc::world::CollisionGroups::default();
        let it = self.world.collider_world().interferences_with_ray(&ray, &g);

        let first = it.filter(|x| x.0.name() != "player").min_by(|a, b| {
            a.1.toi
                .partial_cmp(&b.1.toi)
                .unwrap_or(std::cmp::Ordering::Equal)
        })?;
        let pos = ro + first.1.toi * rd;

        if first.1.toi <= max_t {
            Some(Intersection {
                pos,
                cell: (pos - 0.1 * first.1.normal).map(|x| x as i32 - if x < 0.0 { 1 } else { 0 }),
                t: first.1.toi,
                normal: first.1.normal,
            })
        } else {
            None
        }
    }
}
