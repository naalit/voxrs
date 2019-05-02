#[macro_use]
extern crate glium;
extern crate glm;
extern crate stopwatch;
extern crate rayon;

use glm::*;

// mod octree;
// mod terrain;
// mod chunk;
mod grid;

#[derive(Copy, Clone)]
struct Vertex {
    pos: [f32; 2],
}
implement_vertex!(Vertex, pos);

fn vert(x: f32, y: f32) -> Vertex {
    Vertex { pos: [x, y] }
}

const MOVE_SPEED: f32 = 0.01;

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

    use std::fs::File;
    use std::io::Read;
    let mut vfile = File::open("src/vert.glsl").unwrap();
    let mut vshader = String::new();
    vfile.read_to_string(&mut vshader).unwrap();

    let mut ffile = File::open("src/frag.glsl").unwrap();
    let mut fshader = String::new();
    ffile.read_to_string(&mut fshader).unwrap();

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

    let chunks = grid::generate();
    let chunks = grid::combine(chunks);
    let chunk_buffer = glium::texture::texture3d::Texture3d::with_format(&display, chunks, glium::texture::UncompressedFloatFormat::U16, glium::texture::MipmapsOption::NoMipmap).unwrap();
    println!("{:?}", chunk_buffer.get_internal_format());

    let mut last = timer.elapsed_ms();
    let mut camera_pos = vec3(0.0,10.0,0.0);
    while !closed {
        let cur = timer.elapsed_ms();
        let delta = cur - last;
        println!("FPS: {}", 1000 / delta.max(1));
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
        let camera_up = vec3(0.0, 1.0, 0.0);
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
                   chunks: &chunk_buffer,
                   chunk_origin: [0.0,0.0,0.0_f32],
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
                },
                glutin::WindowEvent::KeyboardInput { input, .. } => {
                    if let glutin::ElementState::Pressed = input.state {
                        match input.virtual_keycode {
                            Some(glutin::VirtualKeyCode::Comma) => {
                                camera_pos = camera_pos + vec3(1.0, 0.0, 1.0) * camera_dir * (delta as f64 * MOVE_SPEED as f64) as f32;
                            },
                            Some(glutin::VirtualKeyCode::O) => {
                                camera_pos = camera_pos - vec3(1.0, 0.0, 1.0) * camera_dir * (delta as f64 * MOVE_SPEED as f64) as f32;
                            },
                            Some(glutin::VirtualKeyCode::Space) => {
                                camera_pos = camera_pos + vec3(0.0, 1.0, 0.0) * (delta as f64 * MOVE_SPEED as f64) as f32;
                            },
                            Some(glutin::VirtualKeyCode::P) => {
                                camera_pos = camera_pos - vec3(0.0, 1.0, 0.0) * (delta as f64 * MOVE_SPEED as f64) as f32;
                            }
                            _ => (),
                        }
                    }
                },
                _ => (),
            },
            /*glutin::Event::DeviceEvent { event, .. } => match event {
                glutin::DeviceEvent::MouseMotion { delta } => mouse = [mouse[0] + delta.0 as f32, mouse[1] + delta.1 as f32, mouse[2] + delta.0 as f32, mouse[3] + delta.1 as f32],
                _ => (),
            },*/
            _ => (),
        });
    }
}
