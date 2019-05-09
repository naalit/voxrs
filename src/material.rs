use enum_iterator::IntoEnumIterator;

#[derive(IntoEnumIterator)]
pub enum Material {
    Air = 0,
    Stone,
    Grass,
    Sand,
    Water,
}
#[derive(Copy,Clone)]
pub struct MatData {
    pub color: [f32; 3],
    pub roughness: f32,
}

implement_uniform_block!(MatData, color, roughness);

impl Material {
    // pub fn num_var() -> usize { Material::Grass as usize }
    pub fn mat_data(&self) -> MatData {
        match self {
            Material::Stone => MatData { color: [0.4; 3], roughness: 0.2 },
            Material::Grass => MatData { color: [0.3,0.7,0.5], roughness: 0.8 },
            Material::Sand => MatData { color: [0.9,0.7,0.6], roughness: 0.6 },
            Material::Water => MatData { color: [0.2,0.4,0.9], roughness: 0.05 },
            Material::Air => MatData { color: [0.0; 3], roughness: 1.0 },
        }
    }
}
