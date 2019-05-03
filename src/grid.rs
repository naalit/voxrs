use super::glm::*;

// Should be a power of 2
pub const CHUNK_SIZE: usize = 16;
// This is in a 'diameter'
pub const CHUNK_NUM: usize = 16;

pub type Block = u16;
pub type Chunk = [[[Block; CHUNK_SIZE]; CHUNK_SIZE]; CHUNK_SIZE];
type World = Vec<Vec<Vec<Chunk>>>;
type WorldU = Vec<Vec<Vec<u16>>>;

pub fn combine(w: World) -> WorldU {
    let mut c = vec![vec![vec![0; CHUNK_SIZE*CHUNK_NUM]; CHUNK_SIZE*CHUNK_NUM]; CHUNK_SIZE*CHUNK_NUM];
    for (x, row_x) in c.iter_mut().enumerate() {
        for (y, row_y) in row_x.iter_mut().enumerate() {
            for (z, b) in row_y.iter_mut().enumerate() {
                *b = w[x/CHUNK_SIZE][y/CHUNK_SIZE][z/CHUNK_SIZE][x%CHUNK_SIZE][y%CHUNK_SIZE][z%CHUNK_SIZE];
            }
        }
    }
    c
}

pub fn generate() -> World {
    let mut c = vec![vec![vec![[[[0; CHUNK_SIZE]; CHUNK_SIZE]; CHUNK_SIZE]; CHUNK_NUM]; CHUNK_NUM]; CHUNK_NUM];
    let rad = CHUNK_NUM as i32 / 2;
    for (x, row_x) in c.iter_mut().enumerate() {
        for (y, row_y) in row_x.iter_mut().enumerate() {
            for (z, b) in row_y.iter_mut().enumerate() {
                *b = gen_chunk(
                    ivec3(x as i32,y as i32,z as i32) - rad
                );
            }
        }
    }
    c
}

pub fn gen_chunk(loc: Vector3<i32>) -> Chunk {
    let mut c = [[[0; CHUNK_SIZE]; CHUNK_SIZE]; CHUNK_SIZE];
    let rad = (CHUNK_SIZE as f32) / 2.0;
    //if loc == ivec3(0,0,0) { println!("Why is zero?"); }
    for (x, row_x) in c.iter_mut().enumerate() {
        for (y, row_y) in row_x.iter_mut().enumerate() {
            for (z, b) in row_y.iter_mut().enumerate() {
                *b = gen_block(
                    to_vec3(loc) * (CHUNK_SIZE as f32)
                        + 0.5
                        + vec3((x as f32) - rad, (y as f32) - rad, (z as f32) - rad),
                );
            }
        }
    }
    c
}

fn gen_block(loc: Vector3<f32>) -> Block {
    let h = height(vec2(loc.x, loc.z));
    if loc.y <= h {
        1
    } else {
        0
    }
}

// from https://www.shadertoy.com/view/4djSRW
fn hash(p: Vec3) -> f32 {
	let mut p3  = fract(p * 0.1031);
    p3 = p3 + dot(p3, vec3(p3.y,p3.z,p3.x) + 19.19);
    return fract((p3.x + p3.y) * p3.z);
}

// from https://www.shadertoy.com/view/4sfGzS
fn noise3(x: Vec3) -> f32 {
    let p = floor(x);
    let mut f = fract(x);
    f = f*f*(-f*2.0+3.0);

    return mix(mix(mix( hash(p+vec3(0.0,0.0,0.0)),
                        hash(p+vec3(1.0,0.0,0.0)),f.x),
                   mix( hash(p+vec3(0.0,1.0,0.0)),
                        hash(p+vec3(1.0,1.0,0.0)),f.x),f.y),
               mix(mix( hash(p+vec3(0.0,0.0,1.0)),
                        hash(p+vec3(1.0,0.0,1.0)),f.x),
                   mix( hash(p+vec3(0.0,1.0,1.0)),
                        hash(p+vec3(1.0,1.0,1.0)),f.x),f.y),f.z);
}

fn noise(pos: Vec3) -> f32 {
    let m: Mat3 = mat3( 0.00,  0.80,  0.60,
                        -0.80,  0.36, -0.48,
                        -0.60, -0.48,  0.64 );
    let mut q = pos*8.0;
    let mut f  = 0.5000*noise3( q ); q = m*q*2.01;
    f += 0.2500*noise3( q );
    q = m*q*2.02;
    f += 0.1250*noise3( q );
    q = m*q*2.03;
    f += 0.0625*noise3( q );
    // q = m*q*2.01;
    f
}

fn noise2(pos: Vec2) -> f32 {
    noise(vec3(pos.x,pos.y,pos.x))
}

fn height(loc: Vector2<f32>) -> f32 {
    noise2(loc*0.001) * 2.0
    //sin(loc.x) * cos(loc.y) * 4.0
}
