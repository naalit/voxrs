use nphysics3d::object::DefaultBodyHandle;
use nphysics3d::object::BodyPartHandle;
use crate::common::*;

use nphysics3d::object::{DefaultBodySet, DefaultColliderSet};
use nphysics3d::force_generator::DefaultForceGeneratorSet;
use nphysics3d::joint::DefaultJointConstraintSet;
use nphysics3d::world::{DefaultMechanicalWorld, DefaultGeometricalWorld};
use nphysics3d as np;

pub struct Physics {
    pub geom: DefaultGeometricalWorld<f32>,
    pub mech: DefaultMechanicalWorld<f32>,

    pub bodies: DefaultBodySet<f32>,
    pub colliders: DefaultColliderSet<f32>,
    pub constraints: DefaultJointConstraintSet<f32>,
    pub generators: DefaultForceGeneratorSet<f32>,

    pub ground: BodyPartHandle<DefaultBodyHandle>,
}

impl Physics {
    pub fn new() -> Self {
        let mut mech = DefaultMechanicalWorld::new(Vector3::new(0.0, -9.81, 0.0));
        let mut geom = DefaultGeometricalWorld::new();

        let mut bodies = DefaultBodySet::new();
        let mut colliders = DefaultColliderSet::new();
        let mut constraints = DefaultJointConstraintSet::new();
        let mut generators = DefaultForceGeneratorSet::new();

        let ground = bodies.insert(np::object::Ground::new());
        let ground = BodyPartHandle(ground, 0);

        Physics {
            geom,
            mech,
            bodies,
            colliders,
            constraints,
            generators,
            ground,
        }
    }

    pub fn step(&mut self) {
        self.mech.step(
            &mut self.geom,
            &mut self.bodies,
            &mut self.colliders,
            &mut self.constraints,
            &mut self.generators,
        );
    }
}
