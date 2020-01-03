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

pub struct KeyCodes {
    forward: u32,
    left: u32,
    back: u32,
    right: u32,

    up: u32,
    down: u32,
}

pub const DEFAULT_KEY_CODES: KeyCodes = KeyCodes {
    forward: 17, // W
    left: 30, // A
    back: 31, // S
    right: 32, // D

    up: 56, // Space
    down: 42, // LShift
};
