use glium::glutin::*;
use num_traits::FromPrimitive;
use num_derive::FromPrimitive;

#[derive(FromPrimitive)]
pub enum KeyPress {
    Forward = 17, // W
    Left = 30, // A
    Back = 31, // S
    Right = 32, // D

    Up = 56, // Space
    Down = 42, // LShift

    Nothing = 0,
}
