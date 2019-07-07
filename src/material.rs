use enum_iterator::IntoEnumIterator;

#[derive(IntoEnumIterator, PartialEq, Clone, Copy, Debug)]
#[repr(u16)]
pub enum Material {
   Air = 0,
   Stone,
   Grass,
   Dirt,
   Water = 4,
   Sand,
}
#[derive(Copy, Clone)]
pub struct MatData {
   pub color: [f32; 3],
   pub roughness: f32,
   pub trans: f32,
   pub metal: f32,
   pub ior: f32,
   pub nothing: f32, // Just buffer to pack it in right
}
implement_uniform_block!(MatData, color, roughness, trans, metal, ior, nothing);

impl Material {
   pub fn mat_data(&self) -> MatData {
       match self {
           Material::Stone => MatData {
               color: [0.4; 3],
               roughness: 0.2,
               trans: 0.0,
               metal: 0.0,
               ior: 1.45,
               nothing: 0.0,
           },
           Material::Grass => MatData {
               color: [0.3, 0.7, 0.5],
               roughness: 0.8,
               trans: 0.0,
               metal: 0.0,
               ior: 1.45,
               nothing: 0.0,
           },
           Material::Dirt => MatData {
               color: [0.4, 0.3, 0.3],
               roughness: 0.9,
               trans: 0.0,
               metal: 0.0,
               ior: 1.45,
               nothing: 0.0,
           },
           Material::Sand => MatData {
               color: [0.9, 0.7, 0.6],
               roughness: 0.6,
               trans: 0.0,
               metal: 0.0,
               ior: 1.45,
               nothing: 0.0,
           },
           Material::Water => MatData {
               color: [0.2, 0.4, 0.9],
               roughness: 0.05,
               trans: 1.0,
               metal: 0.0,
               ior: 1.33,
               nothing: 0.0,
           },
           Material::Air => MatData {
               color: [0.0; 3],
               roughness: 1.0,
               trans: 1.0,
               metal: 0.0,
               ior: 1.0,
               nothing: 0.0,
           },
       }
   }
}
