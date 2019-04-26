#[macro_use]
extern crate glium;
extern crate stopwatch;

#[derive(Copy, Clone)]
struct Vertex {
    pos: [f32; 2],
    col: [f32; 3],
}
implement_vertex!(Vertex, pos, col);

fn vert(x: f32, y: f32, col: [f32; 3]) -> Vertex {
    Vertex { pos: [x, y], col }
}

fn main() {
    use glium::glutin;
    use glium::Surface;

    let mut events_loop = glutin::EventsLoop::new();
    let wb = glutin::WindowBuilder::new()
        .with_title("Vox.rs");
    let cb = glutin::ContextBuilder::new().with_vsync(true);
    let display = glium::Display::new(wb, cb, &events_loop).unwrap();
    display.gl_window().window().set_cursor(glutin::MouseCursor::Crosshair);//.unwrap();

    let vertexes = vec![
        vert(-3.0, -3.0, [0.0, 0.0, 1.0]),
        vert(3.0, -3.0, [1.0, 0.0, 0.0]),
        vert(0.0, 3.0, [0.0, 1.0, 0.0]),
    ];
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

    let mut closed = false;
    let mut mouse = [0.0; 4];
    let mut m_down = false;
    while !closed {
        let mut target = display.draw();
        target.clear_color(0.0, 0.0, 1.0, 1.0);

        let res = target.get_dimensions();
        let res: [f32; 2] = [res.0 as f32, res.1 as f32];
        target.draw(&vbuff, &indices, &program, &uniform! { iTime: timer.elapsed_ms() as f32 / 1000.0, iResolution: res, iMouse: mouse }, &Default::default()).unwrap();

        target.finish().unwrap();

        events_loop.poll_events(|event| match event {
            glutin::Event::WindowEvent { event, .. } => match event {
                glutin::WindowEvent::CloseRequested => closed = true,
                glutin::WindowEvent::MouseInput { state, .. } => match state {
                    glutin::ElementState::Pressed => m_down = true,
                    glutin::ElementState::Released => m_down = false,
                },
                glutin::WindowEvent::CursorMoved { position, .. } => {
                    mouse = [
                        if m_down { position.x as f32 } else { mouse[0] },
                        if m_down { position.y as f32 } else { mouse[1] },
                        position.x as f32,
                        position.y as f32,
                    ]
                }
                _ => (),
            },
            /*glutin::Event::DeviceEvent { event, .. } => match event {
                glutin::DeviceEvent::MouseMotion { delta } => mouse = [mouse[0] + delta.0 as f32, mouse[1] + delta.1 as f32, mouse[2] + delta.0 as f32, mouse[3] + delta.1 as f32],
                _ => (),
            },*/
            _ => (),
        });
        //std::thread::sleep(std::time::Duration::from_millis(100));
    }
}
