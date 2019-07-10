use crate::common::*;
use crate::num_traits::One;
use glium::*;
use nalgebra::{Translation, UnitQuaternion};
use std::rc::Rc;

#[derive(Copy, Clone)]
pub struct Vertex {
    pos: [f32; 3],
    nor: [f32; 3],
    mat: u32,
}

implement_vertex!(Vertex, pos, nor, mat);

pub fn vert(p: Vec3, n: Vec3, m: Material) -> Vertex {
    Vertex {
        pos: p.into(),
        nor: n.into(),
        mat: m as u32,
    }
}

pub struct Mesh {
    empty: bool,
    vbuff: Option<Rc<VertexBuffer<Vertex>>>,
    model_mat: [[f32; 4]; 4],
}

impl Mesh {
    pub fn new(display: &Display, verts: Vec<Vertex>, loc: Vec3, rot: Vec3) -> Self {
        let empty = verts.len() == 0;
        if !empty {
            println!("Mesh length: {}", verts.len());
        }
        let model = Isometry3::from_parts(
            Translation::from(loc),
            UnitQuaternion::from_euler_angles(rot.x, rot.y, rot.z),
        );

        let vbuff = if empty {
            None
        } else {
            Some(Rc::new(VertexBuffer::new(display, &verts).unwrap()))
        };
        let model_mat: [[f32; 4]; 4] = *model.to_homogeneous().as_ref();

        Mesh {
            empty,
            vbuff,
            model_mat,
        }
    }

    pub fn draw<T: glium::uniforms::AsUniformValue, R: glium::uniforms::Uniforms>(
        &self,
        frame: &mut impl Surface,
        program: &Program,
        params: &DrawParameters,
        uniforms: glium::uniforms::UniformsStorage<'_, T, R>,
    ) {
        if !self.empty {
            frame
                .draw(
                    self.vbuff.clone().unwrap().as_ref(),
                    &glium::index::NoIndices(glium::index::PrimitiveType::TrianglesList),
                    program,
                    &uniforms.add("model", self.model_mat),
                    params,
                )
                .unwrap();
        }
    }
}

/// Returns the face normal and thus direction to step in
fn dir(idx: i32) -> Vec3 {
    match idx {
        0 => Vec3::new(1.0, 0.0, 0.0),
        1 => Vec3::new(-1.0, 0.0, 0.0),
        2 => Vec3::new(0.0, 1.0, 0.0),
        3 => Vec3::new(0.0, -1.0, 0.0),
        4 => Vec3::new(0.0, 0.0, 1.0),
        5 => Vec3::new(0.0, 0.0, -1.0),
        _ => panic!("Error: {} is not a valid index", idx),
    }
}

/// Generates the vertices representing one face quad
fn face(idx: i32, p: Vec3, mat: Material) -> Vec<Vertex> {
    let dir = dir(idx); // Also the normal
    let m = dir.map(|x| if x == 0.0 { 1.0 } else { 0.0 });
    /*
    `m` is 1 in the two directions that dir is 0. So, by multiplying 1 and -1 with m and adding to dir, we get corners of the face.
    We need to pick the right vec3, though, so each combination of 2 elements makes a valid face.
    Combinations:
    1 1 // x, y
    1 0
    0 0
    1 1
    0 0
    0 1
    ---
    1 0 // y, z
    0 1
    0 0
    1 1
    0 1
    1 0
    ---
    1 0 // x, z
    1 1
    0 0
    1 1
    0 1
    0 0
    */

    [
        Vec3::new(0.5, 0.5, -0.5),
        Vec3::new(0.5, -0.5, 0.5),
        Vec3::new(-0.5, -0.5, -0.5),
        Vec3::new(0.5, 0.5, 0.5),
        Vec3::new(-0.5, -0.5, 0.5),
        Vec3::new(-0.5, 0.5, -0.5),
    ]
    .into_iter()
    .map(|x| {
        vert(
            x.component_mul(&m) + dir * 0.5 + p.add_scalar(0.5),
            dir,
            mat,
        )
    })
    .collect()
}

/// This is just naive meshing with culling of interior faces within a chunk
/// TODO greedy meshing
pub fn mesh(grid: &Chunk) -> Vec<Vertex> {
    let mut vertices = Vec::new();
    let lens = IVec3::new(
        grid.len() as i32,
        grid[0].len() as i32,
        grid[0][0].len() as i32,
    );
    for (x, column) in grid.iter().enumerate() {
        for (y, slice) in column.iter().enumerate() {
            for (z, block) in slice.iter().enumerate() {
                if *block != Material::Air {
                    let p = Vec3::new(x as f32, y as f32, z as f32);
                    for i in 0..6 {
                        let dir = dir(i);
                        let n = dir.map(|x| x as i32) + IVec3::new(x as i32, y as i32, z as i32);
                        if n.min() < 0
                            || n.x >= lens.x
                            || n.y >= lens.y
                            || n.z >= lens.z
                            || grid[n.x as usize][n.y as usize][n.z as usize] == Material::Air
                        {
                            vertices.append(&mut face(i, p, *block));
                        }
                    }
                }
            }
        }
    }
    vertices
}
