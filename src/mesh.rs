use crate::common::*;
use glium::*;
use nalgebra::{Translation, UnitQuaternion};
use std::rc::Rc;
use std::sync::Arc;
use std::sync::RwLock;

#[derive(Copy, Clone)]
pub struct Vertex {
    pub pos: [f32; 3],
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
    empty_p2: bool, // For phase 2, where we draw transparent things
    vbuff: Option<Rc<VertexBuffer<Vertex>>>,
    vbuff_p2: Option<Rc<VertexBuffer<Vertex>>>,
    model: Isometry3<f32>,
    c: f32,
    model_mat: [[f32; 4]; 4],
}

impl Mesh {
    pub fn new(
        display: &Display,
        verts: Vec<Vertex>,
        verts_p2: Vec<Vertex>,
        loc: Vec3,
        rot: Vec3,
        new: bool,
    ) -> Self {
        let c = if new { 2.0 } else { 0.0 };

        let empty = verts.is_empty();
        let empty_p2 = verts_p2.is_empty();

        let model = Isometry3::from_parts(
            Translation::from(loc),
            UnitQuaternion::from_euler_angles(rot.x, rot.y, rot.z),
        );

        let vbuff = if empty {
            None
        } else {
            Some(Rc::new(VertexBuffer::new(display, &verts).unwrap()))
        };
        let vbuff_p2 = if empty_p2 {
            None
        } else {
            Some(Rc::new(VertexBuffer::new(display, &verts_p2).unwrap()))
        };

        let model_mat: [[f32; 4]; 4] = *model.to_homogeneous().as_ref();

        Mesh {
            empty,
            empty_p2,
            vbuff,
            vbuff_p2,
            c,
            model,
            model_mat,
        }
    }

    pub fn draw<T: glium::uniforms::AsUniformValue, R: glium::uniforms::Uniforms>(
        &mut self,
        frame: &mut impl Surface,
        program: &Program,
        params: &DrawParameters,
        uniforms: glium::uniforms::UniformsStorage<'_, T, R>,
    ) {
        if !self.empty {
            let mut matc = self.model;
            matc.translation.vector.y -= self.c;
            let matc: [[f32; 4]; 4] = *matc.to_homogeneous().as_ref();
            if self.c > 0.0 {
                self.c -= 0.2;
            }
            frame
                .draw(
                    self.vbuff.clone().unwrap().as_ref(),
                    &glium::index::NoIndices(glium::index::PrimitiveType::TrianglesList),
                    program,
                    &uniforms.add("model", matc),
                    params,
                )
                .unwrap();
        }
    }

    pub fn draw_p2<T: glium::uniforms::AsUniformValue, R: glium::uniforms::Uniforms>(
        &mut self,
        frame: &mut impl Surface,
        program: &Program,
        params: &DrawParameters,
        uniforms: glium::uniforms::UniformsStorage<'_, T, R>,
    ) {
        let mut matc = self.model;
        matc.translation.vector.y -= self.c;
        let matc: [[f32; 4]; 4] = *matc.to_homogeneous().as_ref();
        if self.c > 0.0 && self.empty { // Only do this once
            self.c -= 0.2;
        }
        if !self.empty_p2 {
            frame
                .draw(
                    self.vbuff_p2.clone().unwrap().as_ref(),
                    &glium::index::NoIndices(glium::index::PrimitiveType::TrianglesList),
                    program,
                    &uniforms.add("model", matc),
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

pub fn neighbors(idx: IVec3) -> Vec<IVec3> {
    [
        -IVec3::x(),
        IVec3::x(),
        -IVec3::y(),
        IVec3::y(),
        -IVec3::z(),
        IVec3::z(),
    ]
    .iter()
    .map(|x| idx + x)
    .collect()
}

pub fn neighbor_axis(
    neighbors: &[Arc<RwLock<Chunk>>],
    axis: usize,
) -> (Arc<RwLock<Chunk>>, Arc<RwLock<Chunk>>) {
    match axis {
        0 => (neighbors[0].clone(), neighbors[1].clone()),
        1 => (neighbors[2].clone(), neighbors[3].clone()),
        2 => (neighbors[4].clone(), neighbors[5].clone()),
        _ => panic!("Unknown axis: {}", axis),
    }
}

use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize)]
pub enum Mesher {
    Culled,
    Greedy,
}

impl Mesher {
    /// `neighbors` is [-x, +x, -y, +y, -z, +z]
    pub fn mesh(&self, grid: &Chunk, neighbors: Vec<Arc<RwLock<Chunk>>>, phase2: bool) -> Vec<Vertex> {
        match self {
            Mesher::Culled => culled(grid, neighbors, phase2),
            Mesher::Greedy => greedy(grid, neighbors, phase2),
        }
    }
}

/// This is just naive meshing with culling of interior faces within a chunk
fn culled(grid: &Chunk, neighbors: Vec<Arc<RwLock<Chunk>>>, phase2: bool) -> Vec<Vertex> {
    let mut vertices = Vec::new();

    // Sweep on all three axes
    for d in 0..3 {
        // `d` is the main axis, `u` and `v` are the other two
        let u = (d + 1) % 3;
        let v = (d + 2) % 3;

        let mut normal = Vec3::zeros();
        normal[d] = 1.0;

        let fb = neighbor_axis(&neighbors, d);

        // The faces that need to be drawn
        let mut culled = grid.cull_faces(d, (&fb.0.read().unwrap(), &fb.1.read().unwrap()), phase2);
        if culled.is_empty() { continue; }

        // The actual sweeping
        for d_i in 0..=CHUNK_SIZE as usize {
            let culled = &mut culled[d_i];
            // Generate mesh
            for u_i in 0..CHUNK_SIZE as usize {
                for v_i in 0..CHUNK_SIZE as usize {
                    let b = culled[u_i][v_i];
                    if b != Material::Air {
                        // Add this face to the mesh
                        let left = (u_i, v_i);
                        let right = (u_i + 1, v_i + 1);

                        // Generate vertices

                        // Bottom left
                        let mut vleft = Vec3::zeros();
                        vleft[d] = d_i as f32;
                        vleft[u] = left.0 as f32;
                        vleft[v] = left.1 as f32;
                        let vleft = vert(vleft, normal, b);

                        // Top left
                        let mut vmid = Vec3::zeros();
                        vmid[d] = d_i as f32;
                        vmid[u] = right.0 as f32;
                        vmid[v] = left.1 as f32;
                        let vmid = vert(vmid, normal, b);

                        // Top right
                        let mut vright = Vec3::zeros();
                        vright[d] = d_i as f32;
                        vright[u] = right.0 as f32;
                        vright[v] = right.1 as f32;
                        let vright = vert(vright, normal, b);

                        // Bottom right
                        let mut vend = Vec3::zeros();
                        vend[d] = d_i as f32;
                        vend[u] = left.0 as f32;
                        vend[v] = right.1 as f32;
                        let vend = vert(vend, normal, b);

                        // Triangle 1
                        vertices.push(vleft);
                        vertices.push(vmid);
                        vertices.push(vright);

                        // Triangle 2
                        vertices.push(vleft);
                        vertices.push(vright);
                        vertices.push(vend);
                    }
                }
            }
        }
    }

    vertices
}

/// Greedy meshing as in https://0fps.net/2012/06/30/meshing-in-a-minecraft-game/
fn greedy(grid: &Chunk, neighbors: Vec<Arc<RwLock<Chunk>>>, phase2: bool) -> Vec<Vertex> {
    let mut vertices = Vec::new();

    // Sweep on all three axes
    for d in 0..3 {
        // `d` is the main axis, `u` and `v` are the other two
        let u = (d + 1) % 3;
        let v = (d + 2) % 3;

        let mut normal = Vec3::zeros();
        normal[d] = 1.0;

        let fb = neighbor_axis(&neighbors, d);

        // The faces that need to be drawn
        let mut culled = grid.cull_faces(d, (&fb.0.read().unwrap(), &fb.1.read().unwrap()), phase2);
        if culled.is_empty() { continue; }

        // The actual sweeping
        for d_i in 0..=CHUNK_SIZE as usize {
            let culled = &mut culled[d_i];
            // Generate mesh
            for u_i in 0..CHUNK_SIZE as usize {
                for v_i in 0..CHUNK_SIZE as usize {
                    let b = culled[u_i][v_i];
                    if b != Material::Air {
                        // Add this face to the mesh, with any others that are adjacent
                        let left = (u_i, v_i);
                        let mut right = (u_i + 1, v_i + 1);

                        // Add to u
                        for u_i in (u_i + 1)..CHUNK_SIZE as usize {
                            if culled[u_i][v_i] == b {
                                right.0 += 1;

                                // We don't need to mesh this one anymore
                                culled[u_i][v_i] = Material::Air;
                            } else {
                                break;
                            }
                        }

                        // Add to v
                        for v_i in (v_i + 1)..CHUNK_SIZE as usize {
                            // Sweep across the whole u extent of the current quad to make sure we can extend the whole thing
                            if (left.0..right.0)
                                .all(|u_i| culled[u_i][v_i] == b)
                            {
                                right.1 += 1;

                                // We don't need to mesh this whole line anymore
                                (left.0..right.0).for_each(|u_i| {
                                    culled[u_i][v_i] = Material::Air
                                });
                            } else {
                                break;
                            }
                        }

                        // Generate vertices

                        // Bottom left
                        let mut vleft = Vec3::zeros();
                        vleft[d] = d_i as f32;
                        vleft[u] = left.0 as f32;
                        vleft[v] = left.1 as f32;
                        let vleft = vert(vleft, normal, b);

                        // Top left
                        let mut vmid = Vec3::zeros();
                        vmid[d] = d_i as f32;
                        vmid[u] = right.0 as f32;
                        vmid[v] = left.1 as f32;
                        let vmid = vert(vmid, normal, b);

                        // Top right
                        let mut vright = Vec3::zeros();
                        vright[d] = d_i as f32;
                        vright[u] = right.0 as f32;
                        vright[v] = right.1 as f32;
                        let vright = vert(vright, normal, b);

                        // Bottom right
                        let mut vend = Vec3::zeros();
                        vend[d] = d_i as f32;
                        vend[u] = left.0 as f32;
                        vend[v] = right.1 as f32;
                        let vend = vert(vend, normal, b);

                        // Triangle 1
                        vertices.push(vleft);
                        vertices.push(vmid);
                        vertices.push(vright);

                        // Triangle 2
                        vertices.push(vleft);
                        vertices.push(vright);
                        vertices.push(vend);
                    }
                }
            }
        }
    }
    vertices
}
