use glm::*;

/// The type of the base octree node; leaf bitmask & eight pointers.
/// We use eight pointers instead of one to eight sequential children so that it's easier to add and remove non-leaf children, without having to move nodes, which would be prohibitively expensive.
/// The indices for each of these arrays have bits corresponding to children on the axis `x,y,z`, from most to least significant
#[derive(PartialEq, Debug)]
pub struct Node {
    pub leaf: [bool; 8],
    pub pointer: [usize; 8], // each is actually 23-bit
}

pub type NodeB = [f64; 4];
pub type Octree = Vec<Node>;

/// (The `n` least significant digits of `x`, the rest shifted right `n`)
fn bits(x: usize, n: usize) -> (usize, usize) {
    let b = 2_usize.pow(n as u32 + 1) - 1;
    (x & b, x >> n)
}

impl Node {
    /// Layout:
    /// ```
    /// [
    ///     f64: 8leaf + 8 * 7pointer
    ///     f64: 4 * 16pointer
    ///     f64: 4 * 16pointer
    ///     0.0 // Needed for packing right into the GLSL SSBO
    /// ]
    /// ```
    /// Each 7 pointer bits from the `x` value are left-shifted by 16 and then `|`-ed on
    pub fn uniform(&self) -> NodeB {
        let pieces: Vec<(usize, usize)> = self.pointer.iter().map(|x| bits(*x, 16)).collect();
        let leaf = self.leaf;
        let mut one = (u64::from(leaf[0]) << 63)
            | (u64::from(leaf[1]) << 62)
            | (u64::from(leaf[2]) << 61)
            | (u64::from(leaf[3]) << 60)
            | (u64::from(leaf[4]) << 59)
            | (u64::from(leaf[5]) << 58)
            | (u64::from(leaf[6]) << 57)
            | (u64::from(leaf[7]) << 56);
        for (i, (_, x)) in pieces.iter().enumerate() {
            one |= (*x as u64) << (56 - 7 * (i + 1)); // Enumerate starts at 0, but we want to start at 56-7, so i+1
        }
        let two: u64 = (pieces[0].0 as u64) << 48
            | (pieces[1].0 as u64) << 32
            | (pieces[2].0 as u64) << 16
            | (pieces[3].0 as u64);
        let three: u64 = (pieces[4].0 as u64) << 48
            | (pieces[5].0 as u64) << 32
            | (pieces[6].0 as u64) << 16
            | (pieces[7].0 as u64);
        [
            f64::from_bits(one),
            f64::from_bits(two),
            f64::from_bits(three),
            0.0,
        ]
    }

    // This follows the GLSL equivalent as closely as possible, so it can be tested.
    //  Only update both at once
    pub fn decode(source: Vector3<f64>) -> Node {
        let mut leaf = [false; 8];
        // Reverse order - least significant, most significant
        let mut one = unpackDouble2x32(source.x);
        for i in 0..8 {
            leaf[i] = one.y >= (1 << (31 - i));
            one.y = one.y % (1 << (31 - i));
        }
        let mut pointer: [usize; 8] = [0; 8];
        // Reverse order again
        let two = unpackDouble2x32(source.y);
        let three = unpackDouble2x32(source.z);

        pointer[0] = bitfieldExtract(two.y, 16, 16) as usize; // Most significant
        pointer[1] = bitfieldExtract(two.y, 0, 16) as usize; // Least significant
        pointer[2] = bitfieldExtract(two.x, 16, 16) as usize; // Most significant
        pointer[3] = bitfieldExtract(two.x, 0, 16) as usize; // Least significant

        pointer[4] = bitfieldExtract(three.y, 16, 16) as usize; // Most significant
        pointer[5] = bitfieldExtract(three.y, 0, 16) as usize; // Least significant
        pointer[6] = bitfieldExtract(three.x, 16, 16) as usize; // Most significant
        pointer[7] = bitfieldExtract(three.x, 0, 16) as usize; // Least significant

        pointer[0] |= (bitfieldExtract(one.y, 25 - 8, 7) as usize) << 16; // bits 0-7 are used up
        pointer[1] |= (bitfieldExtract(one.y, 25 - 15, 7) as usize) << 16;
        pointer[2] |= (bitfieldExtract(one.y, 25 - 22, 7) as usize) << 16;

        pointer[3] |= (bitfieldExtract(one.y, 29 - 29, 3) as usize) << 20; // 16+4=20; 29+3=32
        pointer[3] |= (bitfieldExtract(one.x, 28 - 0, 4) as usize) << 16;

        pointer[4] |= (bitfieldExtract(one.x, 25 - 4, 7) as usize) << 16;
        pointer[5] |= (bitfieldExtract(one.x, 25 - 11, 7) as usize) << 16;
        pointer[6] |= (bitfieldExtract(one.x, 25 - 18, 7) as usize) << 16;
        pointer[7] |= (bitfieldExtract(one.x, 25 - 25, 7) as usize) << 16; // 25 + 7 = 32

        Node { leaf, pointer }
    }

    /// Converts between a 3D vector representing the child slot, and the actual index into the `leaf` and `pointer` arrays
    pub fn idx<T: BaseNum>(idx: Vector3<T>) -> usize {
        // Once again, this function closely mirrors the GLSL one for testing
        let mut ret = 0;
        ret |= usize::from(idx.x > T::zero()) << 2;
        ret |= usize::from(idx.y > T::zero()) << 1;
        ret |= usize::from(idx.z > T::zero());
        ret
    }

    /// Converts between a 3D vector representing the child slot, and the actual index into the `leaf` and `pointer` arrays
    pub fn position(idx: usize) -> Vector3<f32> {
        vec3(
            if idx & (1 << 2) > 0 { 1.0 } else { -1.0 },
            if idx & (1 << 1) > 0 { 1.0 } else { -1.0 },
            if idx & 1 > 0 { 1.0 } else { -1.0 },
        )
    }
}

pub fn to_uniform(x: Octree) -> Box<[NodeB]> {
    x.iter()
        .map(|x| x.uniform())
        .collect::<Vec<NodeB>>()
        .into_boxed_slice()
}

// --------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_decode() {
        let n = Node {
            leaf: [false, true, false, false, true, false, true, true],
            // The max value for 23 bits is 8_388_607
            pointer: [
                8_123_043, 5_918_743, 1_483_799, 3_059_482, 34_257, 3_425_789, 0, 923_475,
            ],
        };
        let u = n.uniform();
        assert_eq!(n, Node::decode(dvec3(u[0], u[1], u[2])));
        let encoded: NodeB = [
            // Any sequence of f64 bits should be valid
            18953794.423546,
            345789245897.45235,
            43890235904358.5823497592837589,
            0.0,
        ];
        assert_eq!(
            encoded,
            Node::decode(dvec3(encoded[0], encoded[1], encoded[2])).uniform()
        );
    }

    #[test]
    fn decode_idx() {
        let a_i = 0b101;
        let a_v = vec3(1.0, -1.0, 1.0);
        let b_i = 0b010;
        let b_v = vec3(-1.0, 1.0, -1.0);

        assert_eq!(a_i, Node::idx(a_v));
        assert_eq!(b_i, Node::idx(b_v));
        assert_ne!(Node::idx(a_v), Node::idx(b_v));

        assert_eq!(a_v, Node::position(a_i));
        assert_eq!(b_v, Node::position(b_i));
    }
}
