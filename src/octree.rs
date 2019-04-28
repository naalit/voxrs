use glm::*;

pub struct Node {
    pub leaf: [bool; 8],
    pub pointer: [usize; 8], // each is actually 23-bit
}

pub type Octree = Vec<Node>;

/// (The `n` least significant digits of `x`, the rest shifted right `n`)
fn bits(x: usize, n: usize) -> (usize, usize) {
    let mut b = 0;
    for i in 0..n {
        b += 2_usize.pow(i as u32);
    }
    (x & b, x >> n)
}

impl Node {
    /// Layout:
    /// ```
    /// [
    ///     f64: 8leaf + 8 * 7pointer
    ///     f64: 4 * 16pointer
    ///     f64: 4 * 16pointer
    /// ]
    /// ```
    /// Each 7 pointer bits from the `x` value are left-shifted by 16 and then `|`-ed on
    pub fn uniform(&self) -> [f64; 3] {
        let pieces: Vec<(usize, usize)> = self.pointer.iter().map(|x| bits(*x,16)).collect();
        let leaf = self.leaf;
        let mut one =
            (u64::from(leaf[0]) << 63) |
            (u64::from(leaf[1]) << 62) |
            (u64::from(leaf[2]) << 61) |
            (u64::from(leaf[3]) << 60) |
            (u64::from(leaf[4]) << 59) |
            (u64::from(leaf[5]) << 58) |
            (u64::from(leaf[6]) << 57) |
            (u64::from(leaf[7]) << 56);
        for (i,(_,x)) in pieces.iter().enumerate() {
            one |= (*x as u64) << (56 - 7*(i+1)); // Enumerate starts at 0, but we want to start at 56-7, so i+1
        }
        let two: u64 =
            (pieces[0].0 as u64) << 56 |
            (pieces[1].0 as u64) << 32 |
            (pieces[2].0 as u64) << 16 |
            (pieces[3].0 as u64);
        let three: u64 =
            (pieces[4].0 as u64) << 56 |
            (pieces[5].0 as u64) << 32 |
            (pieces[6].0 as u64) << 16 |
            (pieces[7].0 as u64);
        [
            f64::from_bits(one),
            f64::from_bits(two),
            f64::from_bits(three),
        ]
    }
}

pub fn to_uniform(x: Octree) -> Box<[[f64; 3]]> {
    x.iter().map(|x| x.uniform()).collect::<Vec<[f64; 3]>>().into_boxed_slice()
}

/*
impl Octree {
    pub fn uniform(&self) -> Box<[]> {
        self.iter().map(|x| x.uniform()).collect<>()
    }
}
*/
