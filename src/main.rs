#[macro_use]
extern crate glium;
extern crate glm;
extern crate glsl_include;
extern crate rayon;
extern crate stopwatch;
extern crate noise;
extern crate rand;

use glm::*;
use glsl_include::Context as ShaderContext;

// mod octree;
// mod terrain;
mod chunk;
mod terrain;
mod common;

#[derive(Copy, Clone)]
struct Vertex {
    pos: [f32; 2],
}
implement_vertex!(Vertex, pos);

fn vert(x: f32, y: f32) -> Vertex {
    Vertex { pos: [x, y] }
}

struct FrameState {
    fb: f32, // * camera_dir
    lr: f32, // * right
    ud: f32, // * vec3(0,1,0)
    m: f32,  // Multiplier for movement speed
}

const MOVE_SPEED: f32 = 0.01;

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
    use glium::glutin;
    use glium::Surface;

    let mut events_loop = glutin::EventsLoop::new();
    let wb = glutin::WindowBuilder::new().with_title("Vox.rs");
    let cb = glutin::ContextBuilder::new().with_vsync(true);
    let display = glium::Display::new(wb, cb, &events_loop).unwrap();
    display
        .gl_window()
        .window()
        .set_cursor(glutin::MouseCursor::Crosshair); //.unwrap();

    let vertexes = vec![vert(-3.0, -3.0), vert(3.0, -3.0), vert(0.0, 3.0)];
    let vbuff = glium::VertexBuffer::new(&display, &vertexes).unwrap();
    let indices = glium::index::NoIndices(glium::index::PrimitiveType::TriangleStrip);

    let vshader = shader("vert.glsl".to_string(), &[]);
    let fshader = shader("frag.glsl".to_string(), &["sky.glsl".to_string()]);

    let program = glium::Program::from_source(&display, &vshader, &fshader, None).unwrap();

    let timer = stopwatch::Stopwatch::start_new();
    let initial_time =
        6.0 // 06:00, in minutes
        * 60.0 // Seconds
        ;

    let mut closed = false;
    let mut mouse = vec2(0.0, 0.0);
    let mut m_down = false;
    /*
    let octree = vec![
        octree::Node {
            leaf: [true, true, false, true, true, true, true, true],
            pointer: [0, 0, 1, 0, 1, 0, 0, 0],
        },
        octree::Node {
            leaf: [true; 8],
            pointer: [0, 0, 0, 1, 0, 0, 0, 0],
        },
    ];
    let max_length = 2;*/
    // let octree = terrain::generate();
    // let max_length = octree.len();
    // println!("{}",max_length);
    // let mut octree_buffer: glium::buffer::Buffer<[[f64; 4]]> =
    //     glium::buffer::Buffer::empty_unsized(//empty_unsized_persistent(
    //         &display,
    //         glium::buffer::BufferType::ShaderStorageBuffer,
    //         std::mem::size_of::<[f64; 4]>() * max_length,
    //         glium::buffer::BufferMode::Persistent,
    //     )
    //     .unwrap();
    /*{
        let mut octree_pointer = octree_buffer.map_write();
        octree_pointer.set(0, octree[0].uniform());
        octree_pointer.set(1, octree[1].uniform());
    }*/
    // octree_buffer.write(&octree::to_uniform(octree));

    let mut chunk_m = chunk::ChunkManager::new(&display, ivec3(0, 0, 0));

    let mut last = timer.elapsed_ms();
    let mut camera_pos = vec3(0.0, 10.0, 0.0);
    let mut state = FrameState {
        fb: 0.0,
        lr: 0.0,
        ud: 0.0,
        m: 1.0,
    };
    while !closed {
        chunk_m.update(camera_pos);
        let cur = timer.elapsed_ms();
        let delta = cur - last;
        // println!("FPS: {}", 1000 / delta.max(1));
        last = cur;
        let mut target = display.draw();
        target.clear_color(0.0, 0.0, 1.0, 1.0);

        let res = target.get_dimensions();
        let res = vec2(res.0 as f32, res.1 as f32);
        let r = 12. * mouse.x / res.x;
        let look_at = vec3(
            camera_pos.x + 5.0 * (0.5 * r).sin(),
            camera_pos.y + 6.0 * mouse.y / res.y,
            camera_pos.z + 5.0 * (0.5 * r).cos(),
        );
        //let look_at = vec3(0.0, 23.0, 0.0);
        let camera_dir = normalize(look_at - camera_pos);
        // let camera_dir = vec3(0.0,0.0,1.0);
        let camera_up = vec3(0.0, 1.0, 0.0);
        let right = cross(camera_dir, camera_up);
        target
            .draw(
                &vbuff,
                &indices,
                &program,
                &uniform! {
                   iTime: initial_time + timer.elapsed_ms() as f32 / 1000.0,
                   iResolution: *res.as_array(),
                   iMouse: *mouse.as_array(),
                   cameraPos: *camera_pos.as_array(),
                   cameraDir: *camera_dir.as_array(),
                   cameraUp: *camera_up.as_array(),
                   chunks: chunk_m.chunks_u(),
                   blocks: chunk_m.blocks_u(),
                   chunk_origin: chunk_m.origin_u(),
                   // octree: &octree_buffer,
                   levels: 2,
                },
                &Default::default(),
            )
            .unwrap();

        //std::thread::sleep(std::time::Duration::from_millis(100));
        target.finish().unwrap();

        events_loop.poll_events(|event| match event {
            glutin::Event::WindowEvent { event, .. } => match event {
                glutin::WindowEvent::CloseRequested => closed = true,
                glutin::WindowEvent::MouseInput { state, .. } => match state {
                    glutin::ElementState::Pressed => m_down = true,
                    glutin::ElementState::Released => m_down = false,
                },
                glutin::WindowEvent::CursorMoved { position, .. } => {
                    mouse = vec2(
                        if m_down { position.x as f32 } else { mouse.x },
                        if m_down { position.y as f32 } else { mouse.y },
                    )
                }
                glutin::WindowEvent::KeyboardInput { input, .. } => {
                    if let glutin::ElementState::Released = input.state {
                        match input.virtual_keycode {
                            Some(glutin::VirtualKeyCode::Comma)
                            | Some(glutin::VirtualKeyCode::O) => {
                                state.fb = 0.0;
                            }
                            Some(glutin::VirtualKeyCode::Space)
                            | Some(glutin::VirtualKeyCode::LShift) => {
                                state.ud = 0.0;
                            }
                            Some(glutin::VirtualKeyCode::E) | Some(glutin::VirtualKeyCode::A) => {
                                state.lr = 0.0;
                            }
                            Some(glutin::VirtualKeyCode::LControl) => {
                                state.m = 1.0;
                            }
                            _ => (),
                        }
                    }
                    if let glutin::ElementState::Pressed = input.state {
                        match input.virtual_keycode {
                            Some(glutin::VirtualKeyCode::Comma) => {
                                state.fb = 1.0;
                            }
                            Some(glutin::VirtualKeyCode::O) => {
                                state.fb = -1.0;
                            }
                            Some(glutin::VirtualKeyCode::Space) => {
                                state.ud = 1.0;
                            }
                            Some(glutin::VirtualKeyCode::LShift) => {
                                state.ud = -1.0;
                            }
                            Some(glutin::VirtualKeyCode::E) => {
                                state.lr = 1.0;
                            }
                            Some(glutin::VirtualKeyCode::A) => {
                                state.lr = -1.0;
                            }
                            Some(glutin::VirtualKeyCode::LControl) => {
                                state.m = 4.0;
                            }
                            _ => (),
                        }
                    }
                }
                _ => (),
            },
            /*glutin::Event::DeviceEvent { event, .. } => match event {
                glutin::DeviceEvent::MouseMotion { delta } => mouse = [mouse[0] + delta.0 as f32, mouse[1] + delta.1 as f32, mouse[2] + delta.0 as f32, mouse[3] + delta.1 as f32],
                _ => (),
            },*/
            _ => (),
        });
        camera_pos = camera_pos
            + (vec3(1.0, 0.0, 1.0) * camera_dir * state.fb
                + vec3(0.0, 1.0, 0.0) * state.ud
                + vec3(1.0, 0.0, 1.0) * right * state.lr)
                * state.m
                * (delta as f64 * MOVE_SPEED as f64) as f32;
    }
}
