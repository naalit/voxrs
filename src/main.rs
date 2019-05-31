#[macro_use]
extern crate glium;
extern crate glm;
extern crate glsl_include;
extern crate num_traits;
#[macro_use]
extern crate num_derive;

use glium::glutin;
use glium::Surface;
use glm::*;
use glsl_include::Context as ShaderContext;
use num_traits::identities::*;

mod input;
use input::*;
mod chunk;
use chunk::*;
mod terrain;
use terrain::*;

#[derive(Copy, Clone)]
struct Vertex {
    pos: [f32; 2],
}

implement_vertex!(Vertex, pos);

fn vert(x: f32, y: f32) -> Vertex {
    Vertex { pos: [x, y] }
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

fn main() {
    // Wayland doesn't allow cursor grabbing
    let mut events_loop = glutin::os::unix::EventsLoopExt::new_x11().unwrap();
    let wb = glutin::WindowBuilder::new().with_title("Vox.rs 2");
    let cb = glutin::ContextBuilder::new();
    let display = glium::Display::new(wb, cb, &events_loop).unwrap();

    // A screen-filling triangle is slightly faster than a quad
    let vertices = vec![vert(-3.0, -3.0), vert(3.0, -3.0), vert(0.0, 3.0)];
    let vbuff = glium::VertexBuffer::new(&display, &vertices).unwrap();
    let indices = glium::index::NoIndices(glium::index::PrimitiveType::TriangleStrip);

    let vshader = shader("vert.glsl".to_string(), &[]);
    let fshader = shader(
        "frag.glsl".to_string(),
        &["octree.glsl".to_string(), "shade.glsl".to_string()],
    );
    let program = glium::Program::from_source(&display, &vshader, &fshader, None).unwrap();
/*
    let octree = generate(); //vec![Node{pointer:[0,2,2,2,2,2,2,2]}];
    println!("{}", octree.len());
    let octree_buffer = glium::uniforms::UniformBuffer::empty_unsized(
        &display,
        octree.len() * std::mem::size_of::<Node>(),
    )
    .unwrap();
    octree_buffer.write(octree.as_slice());
*/

    let mut client = ClientData::new(&display);
    client.load_chunks(vec3(0.0,0.0,0.0), //vec![(ivec3(0,0,0),generate(ivec3(1,0,0)))]);
        (0..2).flat_map(|x| (0..2).flat_map(move |y| (0..2).map(move |z| ivec3(x,y,z)))).map(|x| (x, generate(x))).collect());
    println!("{:?}", client.root);
    println!("{:?}", client.map);

    let (mut rx, mut ry) = (0.0, 0.0);
    let mut camera_pos = vec3(0.0, 1.0, 0.0);
    let mut timer = stopwatch::Stopwatch::start_new();

    let mut moving = vec3(0.0, 0.0, 0.0); // vec3(forwards, up, right)

    let mut open = true;
    while open {
        let delta = timer.elapsed_ms() as f64 / 1000.0;
        println!("{:.1} FPS", 1.0/delta);
        timer.restart();
        let mut target = display.draw();
        target.clear_color(0.0, 0.0, 0.0, 1.0);

        let resolution = target.get_dimensions();

        let camera_up = vec3(0.0, 1.0, 0.0);
        let q = glm::ext::rotate(&Matrix4::one(), rx / resolution.0 as f32 * -6.28, camera_up)
            * vec4(0.0, 0.0, 1.0, 1.0);
        let camera_dir = normalize(vec3(q.x, q.y, q.z));
        let camera_right = cross(camera_dir, camera_up);
        let q = glm::ext::rotate(
            &Matrix4::one(),
            ry / resolution.1 as f32 * -6.28,
            camera_right,
        ) * q;
        let camera_dir = normalize(vec3(q.x, q.y, q.z));
        let camera_up = cross(camera_right, camera_dir);

        target
            .draw(
                &vbuff,
                &indices,
                &program,
                &uniform! {
                    resolution: resolution,
                    octree_buffer: client.tree_uniform(),
                    camera_dir: *camera_dir.as_array(),
                    camera_up: *camera_up.as_array(),
                    camera_right: *camera_right.as_array(),
                    camera_pos: *camera_pos.as_array(),
                },
                &Default::default(),
            )
            .unwrap();

        events_loop.poll_events(|event| match event {
            glutin::Event::WindowEvent {
                event: glutin::WindowEvent::CloseRequested,
                ..
            } => open = false,
            glutin::Event::DeviceEvent { event, .. } => match event {
                glutin::DeviceEvent::MouseMotion { delta } => {
                    rx += delta.0 as f32;
                    ry += delta.1 as f32;
                    ry = glm::clamp(
                        ry,
                        -(resolution.1 as f32 * 0.25),
                        resolution.1 as f32 * 0.25,
                    );
                }
                glutin::DeviceEvent::Key(glutin::KeyboardInput {
                    scancode,
                    state: glutin::ElementState::Pressed,
                    ..
                }) => match num_traits::FromPrimitive::from_u32(scancode)
                    .unwrap_or(KeyPress::Nothing)
                {
                    KeyPress::Forward => moving.x = 1.0,
                    KeyPress::Back => moving.x = -1.0,
                    _ => (),
                },
                glutin::DeviceEvent::Key(glutin::KeyboardInput {
                    scancode,
                    state: glutin::ElementState::Released,
                    ..
                }) => match num_traits::FromPrimitive::from_u32(scancode)
                    .unwrap_or(KeyPress::Nothing)
                {
                    KeyPress::Forward | KeyPress::Back => moving.x = 0.0,
                    _ => (),
                },
                _ => (),
            },
            _ => (),
        });
        camera_pos = camera_pos + camera_dir * moving.x * delta as f32;

        target.finish().unwrap();
    }
}
