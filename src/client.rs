use crate::client_aux::*;
use crate::common::*;
use crate::mesh::*;
use crate::physics::*;
use enum_iterator::IntoEnumIterator;
use glium::glutin::*;
use glium::*;
use glsl_include::Context as ShaderContext;
use std::collections::HashMap;
use std::ops::Deref;
use std::sync::mpsc::*;
use std::sync::{Arc, RwLock};

const NUM_CASCADES: usize = 4;
const SHADOW_RES: u32 = 2048 * 2;
const SHADOW_FAC: f32 = 4.0;

use glium_glyph::glyph_brush::{rusttype::Font, Section};
use glium_glyph::GlyphBrush;
#[derive(Clone)]
struct Camera {
    pos: Point3<f32>,
    dir: Vec3,
    rx: f32,
    ry: f32,
    moving: Vec3, // x is forward, y is up, z is right
    config: Arc<ClientConfig>,
}

impl Camera {
    pub fn new(pos: Point3<f32>, config: Arc<ClientConfig>) -> Camera {
        Camera {
            pos,
            dir: Vec3::new(0.0, 0.0, 1.0),
            rx: 0.0,
            ry: 0.0,
            moving: Vec3::new(0.0, 0.0, 0.0),
            config,
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
            }) => match scancode {
                x if x == self.config.keycodes.forward => self.moving.x = 1.0,
                x if x == self.config.keycodes.back => self.moving.x = -1.0,
                x if x == self.config.keycodes.up => {
                    // Jump
                    self.moving.y = 1.0;
                }
                _ => (),
            },
            glutin::DeviceEvent::Key(glutin::KeyboardInput {
                scancode,
                state: glutin::ElementState::Released,
                ..
            }) => {
                if scancode == self.config.keycodes.forward || scancode == self.config.keycodes.back
                {
                    self.moving.x = 0.0;
                }
            }
            _ => (),
        };
    }

    pub fn update(
        &mut self,
        delta: f64,
        resolution: (u32, u32),
        player: &mut np::object::RigidBody<f32>,
    ) {
        player.apply_force(
            0,
            &np::algebra::Force3::new(self.dir * self.moving.x * 50.0, Vec3::zeros()),
            np::algebra::ForceType::Impulse,
            true,
        );
        self.pos = Point3::from(player.position().translation.vector) + Vec3::new(0.0, 0.4, 0.0);
        //        self.pos = self.pos + self.dir * self.moving.x * delta as f32 * 8.0;

        let camera_up = Unit::new_normalize(Vec3::new(0.0, 1.0, 0.0));

        if self.moving.y > 0.0 {
            if player.velocity().as_vector()[1].abs() < 0.01 {
                player.apply_force(
                    0,
                    &np::algebra::Force3::new(Vec3::y() * 500.0, Vec3::zeros()),
                    np::algebra::ForceType::Impulse,
                    true,
                );
            }
            self.moving.y = 0.0;
        }

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
            0.125,
            1024.0,
        );
        let proj_mat =
            proj_mat * na::Matrix4::look_at_rh(&self.pos, &(self.pos + self.dir), &camera_up);
        proj_mat.into()
    }

    fn view_mat(&self) -> na::Matrix4<f32> {
        let camera_up = Vec3::new(0.0, 1.0, 0.0);
        let camera_right = Unit::new_normalize(self.dir.cross(&camera_up));
        let camera_up = camera_right.cross(&self.dir);

        na::Matrix4::look_at_rh(&self.pos, &(self.pos + self.dir), &camera_up)
    }

    pub fn cascades(
        &self,
        num_cascades: usize,
        resolution: (u32, u32),
        sun_dir: Vec3,
    ) -> Vec<na::Matrix4<f32>> {
        // See http://ogldev.atspace.co.uk/www/tutorial49/tutorial49.html

        let aspect = resolution.0 as f32 / resolution.1 as f32;

        let fovy = radians(90.0) * 0.5;
        let tan_fy = fovy.tan();
        let fovx = fovy * aspect;
        let tan_fx = fovx.tan();

        let m = self.view_mat();
        // camera space -> world space
        let mi = m.try_inverse().unwrap();

        (0..num_cascades)
            .map(|i| {
                let near = 0.125 + 1023.875 * (i as f32 / num_cascades as f32).powf(SHADOW_FAC);
                let far = 0.125 + 1023.875 * ((i + 1) as f32 / num_cascades as f32).powf(SHADOW_FAC);
                let x_near = near * tan_fx;
                let x_far = far * tan_fx;
                let y_near = near * tan_fy;
                let y_far = far * tan_fy;

                // The frustum corners
                let n_a = Vec3::new(x_near, y_near, near);
                let n_b = Vec3::new(x_near, -y_near, near);
                let n_c = Vec3::new(-x_near, y_near, near);
                let n_d = Vec3::new(-x_near, -y_near, near);

                let f_a = Vec3::new(x_far, y_far, far);
                let f_b = Vec3::new(x_far, -y_far, far);
                let f_c = Vec3::new(-x_far, y_far, far);
                let f_d = Vec3::new(-x_far, -y_far, far);

                // Transform them to world space
                let v: Vec<_> = vec![n_a, n_b, n_c, n_d, f_a, f_b, f_c, f_d]
                    .into_iter()
                    .map(|x| (mi * na::Vector4::new(x.x, x.y, x.z, 1.0)).scale())
                    .collect();

                let min = v.iter().fold(
                    Vec3::new(10_000_000.0, 10_000_000.0, 10_000_000.0),
                    |x, a| x.zip_map(a, f32::min),
                );
                let max = v.iter().fold(
                    -Vec3::new(10_000_000.0, 10_000_000.0, 10_000_000.0),
                    |x, a| x.zip_map(a, f32::max),
                );

                let center = (min + max) * 0.5;
                let center = Point3::from(center);
                let extent = (min - max).norm() * 0.5;

                // Average point
                let centroid = v.iter().sum::<Vec3>() / 8.0;
                let centroid = Point3::from(centroid);

                let view_up = Vec3::y();
                let view_right = Unit::new_normalize((-sun_dir).cross(&view_up));
                let view_up = view_right.cross(&(-sun_dir));
                // world space -> light space
                let view =
                    na::Matrix4::look_at_rh(&(center + sun_dir * extent), &center, &view_up);

                // To light space
                let v: Vec<_> = v
                    .into_iter()
                    .map(|x| (view * na::Vector4::new(x.x, x.y, x.z, 1.0)).scale())
                    .collect();

                // And get a bounding box for orthographic projection
                let min = v.iter().fold(
                    Vec3::new(10_000_000.0, 10_000_000.0, 10_000_000.0),
                    |x, a| x.zip_map(a, f32::min),
                );
                let max = v.iter().fold(
                    -Vec3::new(10_000_000.0, 10_000_000.0, 10_000_000.0),
                    |x, a| x.zip_map(a, f32::max),
                );

                // TODO this is temporary fix for an annoying bug - it makes the light view square
                let min = Vec3::new(min.min(), min.min(), min.min());
                let max = Vec3::new(max.max(), max.max(), max.max());

                // View-projection matrix from the light's point of view
                let depth_projection =
                    na::Matrix4::new_orthographic(min.x, max.x, min.y, max.y, 0.125, extent * 2.0);

                depth_projection * view
            })
            .collect()
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
    shadow_program: Program,
    gbuff: glium::texture::Texture2d,
    gbuff_depth: glium::texture::depth_texture2d::DepthTexture2d,
    shadow: glium::texture::DepthTexture2dArray,
    quad: VertexBuffer<SimpleVert>,
    mat_buf: glium::uniforms::UniformBuffer<[MatData]>,
    light_buf: glium::uniforms::UniformBuffer<[[[f32; 4]; 4]]>,
}

impl<'a> DrawStuff<'a> {
    fn new(display: &Display, resolution: (u32, u32), config: Arc<ClientConfig>) -> Self {
        let vshader = shader("gbuff.vert".to_string(), &[]);
        let fshader = shader("gbuff.frag".to_string(), &[]);
        let gbuff_program = glium::Program::from_source(display, &vshader, &fshader, None).unwrap();

        let fshader = shader("shadow_map.frag".to_string(), &[]);
        let shadow_program =
            glium::Program::from_source(display, &vshader, &fshader, None).unwrap();

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

        let shadow_res = SHADOW_RES;

        let shadow = glium::texture::DepthTexture2dArray::empty_with_format(
            display,
            glium::texture::DepthFormat::I24,
            glium::texture::MipmapsOption::NoMipmap,
            shadow_res,
            shadow_res,
            NUM_CASCADES as u32,
        )
        .unwrap();

        let light_buf = glium::uniforms::UniformBuffer::empty_unsized_persistent(
            display,
            std::mem::size_of::<[[f32; 4]; 4]>() * NUM_CASCADES,
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
            shadow_program,
            gbuff,
            gbuff_depth,
            shadow,
            quad,
            mat_buf,
            light_buf,
        }
    }
}

pub struct Client<'font, 'p> {
    camera: Camera,
    chunks: HashMap<IVec3, Arc<RwLock<Chunk>>>,
    meshes: HashMap<IVec3, Mesh>,
    colliders: HashMap<IVec3, np::object::DefaultColliderHandle>,
    display: Display,
    aux: (Sender<Message>, Receiver<Option<ClientMessage>>),
    time: f64,
    physics: Physics,
    player_handle: np::object::DefaultBodyHandle,
    player_c_handle: np::object::DefaultColliderHandle,
    config: Arc<ClientConfig>,
    inventory: Vec<(Material, usize)>,
    glyph_brush: GlyphBrush<'font, 'p>,
}

impl<'font, 'p> Client<'font, 'p> {
    pub fn new(
        display: Display,
        config: Arc<ClientConfig>,
        conn: Connection,
        player: Vec3,
    ) -> Self {
        let (to, from_them) = channel();
        let (to_them, from) = channel();
        let two = Arc::clone(&config);
        std::thread::spawn(move || client_aux_thread(conn, (to_them, from_them), player, two));

        let mut physics = Physics::new();

        let player_shape = nc::shape::ShapeHandle::new(nc::shape::Capsule::new(0.65, 0.25));
        let player_handle = np::object::RigidBodyDesc::new()
            .translation(player)
            .mass(90.0) // In kg, and this person is 2 meters tall
            .build();
        let player_handle = physics.bodies.insert(player_handle);
        let player_collider = np::object::ColliderDesc::new(player_shape)
            .margin(0.05)
            .build(np::object::BodyPartHandle(player_handle, 0));
        let player_c_handle = physics.colliders.insert(player_collider);

        let dejavu: &[u8] = include_bytes!("../DejaVuSansMono-Bold.ttf");
        let fonts = vec![Font::from_bytes(dejavu).unwrap()];

        let glyph_brush = GlyphBrush::new(&display, fonts);

        Client {
            camera: Camera::new(player.into(), config.clone()),
            chunks: HashMap::with_capacity(config.game_config.draw_chunks.pow(3) / 2),
            meshes: HashMap::with_capacity(config.game_config.draw_chunks.pow(3) / 2),
            colliders: HashMap::with_capacity(config.game_config.draw_chunks.pow(3) / 2),
            display,
            aux: (to, from),
            time: 0.0,
            physics,
            player_handle,
            player_c_handle,
            config,
            inventory: Vec::new(),
            glyph_brush,
        }
    }

    /// The player position
    pub fn pos(&self) -> Vec3 {
        self.camera.pos.coords
    }

    pub fn display(&self) -> &Display {
        &self.display
    }

    #[allow(clippy::float_cmp)]
    fn draw(
        &mut self,
        target: &mut Frame,
        draw_stuff: &DrawStuff,
        gbuff_fb: &mut glium::framebuffer::SimpleFrameBuffer,
        shadow_fb: &mut [glium::framebuffer::SimpleFrameBuffer],
        sun_dir: Vec3,
        cascades: &[na::Matrix4<f32>],
    ) {
        target.clear_color(0.0, 0.0, 0.0, 1.0);
        target.clear_depth(1.0);

        let resolution = target.get_dimensions();

        let proj_mat: [[f32; 4]; 4] = self.camera.mat(resolution);

        let mut light_mats = Vec::new();

        for i in 0..NUM_CASCADES {
            let light_mat: [[f32; 4]; 4] = cascades[i].into();

            // Draw the shadow map
            shadow_fb[i].clear_color_and_depth((0.0, 0.0, 0.0, 0.0), 1.0);
            for (_loc, mesh) in self.meshes.iter_mut() {
                mesh.draw(
                    &mut shadow_fb[i],
                    &draw_stuff.shadow_program,
                    &draw_stuff.params,
                    uniform! {
                        proj_mat: light_mat,
                    },
                )
            }

            light_mats.push(light_mat);
        }

        draw_stuff.light_buf.write(&light_mats);

        let shadow_map = glium::uniforms::Sampler::new(&draw_stuff.shadow)
            .magnify_filter(glium::uniforms::MagnifySamplerFilter::Linear)
            .minify_filter(glium::uniforms::MinifySamplerFilter::Linear)
            .depth_texture_comparison(Some(glium::uniforms::DepthTextureComparison::LessOrEqual));

        // Draw chunks onto the G-Buffer
        gbuff_fb.clear_color_and_depth((0.0, 0.0, 0.0, 0.0), 1.0);
        for (_loc, mesh) in self.meshes.iter_mut() {
            mesh.draw(
                gbuff_fb,
                &draw_stuff.gbuff_program,
                &draw_stuff.params,
                uniform! {
                    proj_mat: proj_mat,
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
                    light_buf: &draw_stuff.light_buf,
                    sun_dir: <[f32; 3]>::from(sun_dir.into()),
                    gbuff: &draw_stuff.gbuff,
                    shadow_map: shadow_map,
                    resolution: resolution,
                },
                &Default::default(),
            )
            .unwrap();

        // Draw p2 chunks onto the G-Buffer
        for (_loc, mesh) in self.meshes.iter_mut() {
            mesh.draw_p2(
                gbuff_fb,
                &draw_stuff.gbuff_program,
                &draw_stuff.params,
                uniform! {
                    proj_mat: proj_mat,
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
                    proj_mat: proj_mat,light_buf: &draw_stuff.light_buf,
                    light_buf: &draw_stuff.light_buf,
                    sun_dir: <[f32; 3]>::from(sun_dir.into()),
                    gbuff: &draw_stuff.gbuff,
                    shadow_map: shadow_map,
                    resolution: resolution,
                },
                &glium::draw_parameters::DrawParameters {
                    blend: Blend::alpha_blending(),
                    ..Default::default()
                },
            )
            .unwrap();

        let mut start = 40.0;
        for (m, n) in &self.inventory {
            let old_start = start;
            self.glyph_brush.queue(Section {
                text: &format!("{:?} x {}", m, n),
                bounds: (resolution.0 as f32, resolution.1 as f32 / 2.0),
                screen_position: (0.0, old_start),
                scale: glium_glyph::glyph_brush::rusttype::Scale::uniform(if start == 40.0 {
                    start += 10.0;
                    20.0
                } else {
                    16.0
                }),
                ..Section::default()
            });
            start += 20.0;
        }

        self.glyph_brush.draw_queued(&self.display, target);
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

        if !self.meshes.is_empty() {
            self.physics.step(); // TODO: make timestep match delta?
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
                            let old = self.set_block(p.cell, Material::Air);
                            if let Some(x) = self.inventory.iter_mut().find(|x| x.0 == old) {
                                x.1 += 1;
                            } else {
                                self.inventory.push((old, 1));
                            }
                        }
                    }
                    glutin::DeviceEvent::MouseWheel { delta } => {
                        let d = match delta {
                            glutin::MouseScrollDelta::LineDelta(_, v) => v,
                            glutin::MouseScrollDelta::PixelDelta(p) => {
                                p.y as f32 / resolution.1 as f32
                            }
                        };

                        if d > 0.0 {
                            let i = self.inventory.remove(0);
                            self.inventory.push(i);
                        } else {
                            let i = self.inventory.remove(self.inventory.len() - 1);
                            self.inventory.insert(0, i);
                        }
                    }
                    // Right-click
                    glutin::DeviceEvent::Button {
                        button: 3,
                        state: glutin::ElementState::Pressed,
                    } => {
                        if let Some(p) = self.trace(self.pos(), self.camera.dir, 16.0) {
                            let b = p.cell + p.normal.map(|x| x as i32);
                            let iso1 = self
                                .physics
                                .bodies
                                .rigid_body(self.player_handle)
                                .unwrap()
                                .position();
                            let shape1 = self
                                .physics
                                .colliders
                                .get(self.player_c_handle)
                                .unwrap()
                                .shape();
                            let shape2 = nc::shape::Cuboid::new(Vec3::repeat(0.5));
                            let iso2 = na::Isometry3::from_parts(
                                na::Translation::from(b.map(|x| x as f32 + 0.5)),
                                na::UnitQuaternion::new_normalize(na::Quaternion::identity()),
                            );
                            let d = nc::query::distance(iso1, shape1.deref(), &iso2, &shape2);
                            if d > 0.01 && !self.inventory.is_empty() {
                                let m = self.inventory.first_mut().unwrap();
                                m.1 -= 1;
                                let m = *m;
                                if m.1 == 0 {
                                    self.inventory.remove(0);
                                }
                                self.set_block(b, m.0);
                            }
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
            self.physics
                .bodies
                .rigid_body_mut(self.player_handle)
                .unwrap(),
        );
        self.camera = camera;

        if let Ok(chunks) = self.aux.1.try_recv() {
            // Only load chunks once per frame
            self.load_chunks(chunks.unwrap());
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
        let mut shadow_fb: Vec<_> = (0..NUM_CASCADES)
            .map(|i| {
                glium::framebuffer::SimpleFrameBuffer::depth_only(
                    &self.display,
                    draw_stuff.shadow.layer(i as u32).unwrap().main_level(),
                )
                .unwrap()
            })
            .collect();

        let mut timer = stopwatch::Stopwatch::start_new();

        // days / second
        let sun_speed = 1.0 / (24.0 * 60.0); // a day is 24 minutes
        let sun_dir = Vec3::new(
            (self.time * sun_speed * std::f64::consts::PI * 2.0).sin() as f32,
            (self.time * sun_speed * std::f64::consts::PI * 2.0).cos() as f32,
            0.0,
        )
        .normalize();

        let mut cascades = self.camera.cascades(NUM_CASCADES, resolution, sun_dir);

        let mut open = true;
        let mut i = 0;
        while open {
            i += 1;
            let delta = timer.elapsed_ms() as f64 / 1000.0;
            self.time += delta;

            if i % 120 == 0 {
                println!("{} FPS", 1.0 / delta);
            }

            timer.restart();

            let mut target = self.display().draw();

            // days / second
            let sun_speed = 1.0 / (24.0 * 60.0); // a day is 24 minutes
            let sun_dir = Vec3::new(
                (self.time * sun_speed * std::f64::consts::PI * 2.0).sin() as f32,
                (self.time * sun_speed * std::f64::consts::PI * 2.0).cos() as f32,
                0.0,
            )
            .normalize();

            self.draw(&mut target, &draw_stuff, &mut gbuff_fb, &mut shadow_fb, sun_dir, &cascades);

            // Most computation should go after this point, while the GPU is rendering

            open = self.update(&mut evloop, delta);

            cascades = self.camera.cascades(NUM_CASCADES, resolution, sun_dir);

            target.finish().unwrap();
        }

        self.aux.0.send(Message::Leave).unwrap();
        while let Ok(m) = self.aux.1.recv() {
            if m.is_none() {
                break;
            }
        }
    }

    pub fn set_block(&mut self, loc: IVec3, new: Material) -> Material {
        let chunk = world_to_chunk(loc.map(|x| x as f32));
        let in_chunk = in_chunk(loc.map(|x| x as f32));

        let chunk_rc = self.chunks.get(&chunk).unwrap();
        let old = chunk_rc.read().unwrap().block(in_chunk);
        chunk_rc.write().unwrap().set_block(in_chunk, new);

        self.aux.0.send(Message::SetBlock(loc, new)).unwrap();

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

        old
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
            self.physics.colliders.remove(c);
        }

        let verts = self
            .config
            .mesher
            .mesh(&chunk_rc.read().unwrap(), neighbors.clone(), false);
        let verts_p2 = self
            .config
            .mesher
            .mesh(&chunk_rc.read().unwrap(), neighbors, true);

        if !verts.is_empty() {
            let v_physics: Vec<_> = verts.iter().map(|x| na::Point3::from(x.pos)).collect();
            let i_physics: Vec<_> = (0..v_physics.len() / 3)
                .map(|x| na::Point3::new(x * 3, x * 3 + 1, x * 3 + 2))
                .collect();
            let chunk_shape =
                nc::shape::ShapeHandle::new(nc::shape::TriMesh::new(v_physics, i_physics, None));

            let chunk_collider = np::object::ColliderDesc::new(chunk_shape)
                .translation(chunk.map(|x| x as f32) * CHUNK_SIZE)
                .build(self.physics.ground);
            let handle = self.physics.colliders.insert(chunk_collider);
            self.colliders.insert(chunk, handle);
        }

        let mesh = Mesh::new(
            &self.display,
            verts,
            verts_p2,
            chunk.map(|x| x as f32) * CHUNK_SIZE,
            Vec3::new(0.0, 0.0, 0.0),
            false,
        );
        self.meshes.insert(chunk, mesh);

        Some(())
    }

    /// Load and mesh a bunch of chunks at once. Prunes unneeded ones as well.
    pub fn load_chunks(&mut self, chunks: ClientMessage) {
        self.prune_chunks();

        for (i, v, v2, s, c) in chunks {
            // TODO indices
            if let Some(chunk_shape) = s {
                let chunk_collider = np::object::ColliderDesc::new(chunk_shape)
                    .translation(i.map(|x| x as f32) * CHUNK_SIZE)
                    .build(self.physics.ground);
                let handle = self.physics.colliders.insert(chunk_collider);
                self.colliders.insert(i, handle);
            }

            let mesh = Mesh::new(
                &self.display,
                v,
                v2,
                i.map(|x| x as f32) * CHUNK_SIZE,
                Vec3::new(0.0, 0.0, 0.0),
                true,
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
            self.physics.colliders.remove(handle);
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

impl Client<'_, '_> {
    pub fn trace(&self, ro: Vec3, rd: Vec3, max_t: f32) -> Option<Intersection> {
        let ray = nc::query::Ray::new(ro.into(), rd);

        let g = nc::pipeline::object::CollisionGroups::default();
        let it = self
            .physics
            .geom
            .interferences_with_ray(&self.physics.colliders, &ray, &g);

        let first = it.filter(|x| x.0 != self.player_c_handle).min_by(|a, b| {
            a.2.toi
                .partial_cmp(&b.2.toi)
                .unwrap_or(std::cmp::Ordering::Equal)
        })?;
        let pos = ro + first.2.toi * rd;

        if first.2.toi <= max_t {
            Some(Intersection {
                pos,
                cell: (pos - 0.1 * first.2.normal).map(|x| x as i32 - if x < 0.0 { 1 } else { 0 }),
                t: first.2.toi,
                normal: first.2.normal,
            })
        } else {
            None
        }
    }
}
