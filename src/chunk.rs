use super::octree::*;
use glm::*;
use rayon::prelude::*;

// Should be a power of 2
pub const CHUNK_SIZE: usize = 16;
// This is in one direction; a radius, if you will
pub const CHUNK_NUM: i32 = 2;

pub fn collapse(tree: &mut Octree, mut n: Node) -> Node {
    println!("Collapse");
    for i in 0..8 {
        if !n.leaf[i] {
            let p = n.pointer[i];
            let v = tree[p].clone();
            let v = collapse(tree, v);
            if v.leaf == [true; 8] && v.pointer == [0; 8] {
                n.leaf[i] = true;
                n.pointer[i] = 0;
                tree.remove(p);
                tree.par_iter_mut().for_each(|x| {
                    x.pointer.iter_mut().zip(x.leaf.iter()).for_each(|(x, l)| {
                        if !l && *x >= p {
                            *x -= 1
                        }
                    })
                });
            } else if v.leaf == [true; 8] && v.pointer == [1; 8] {
                n.leaf[i] = true;
                n.pointer[i] = 1;
                tree.remove(p);
                tree.par_iter_mut().for_each(|x| {
                    x.pointer.iter_mut().zip(x.leaf.iter()).for_each(|(x, l)| {
                        if !l && *x >= p {
                            *x -= 1
                        }
                    })
                });
            } else {
                tree[p] = v;
            }
        }
    }
    n
}

pub fn combine_trees(mut trees: [Octree; 8]) -> Octree {
    let mut lengths = trees
        .iter()
        .map(|x| x.len())
        .scan(1, |st, x| {
            let old = *st;
            *st += x;
            Some(old)
        })
        .collect::<Vec<usize>>()
        .into_iter();
    lengths.clone().zip(trees.iter_mut()).for_each(|(i, x)| {
        x.iter_mut()
            .for_each(|y| y.pointer.iter_mut().for_each(|z| *z = *z + i))
    });
    let root = Node {
        leaf: [false; 8],
        pointer: [
            lengths.next().unwrap(),
            lengths.next().unwrap(),
            lengths.next().unwrap(),
            lengths.next().unwrap(),
            lengths.next().unwrap(),
            lengths.next().unwrap(),
            lengths.next().unwrap(),
            lengths.next().unwrap(),
        ],
    };
    let mut tree: Octree = vec![root];
    tree.append(&mut trees[0]);
    tree.append(&mut trees[1]);
    tree.append(&mut trees[2]);
    tree.append(&mut trees[3]);
    tree.append(&mut trees[4]);
    tree.append(&mut trees[5]);
    tree.append(&mut trees[6]);
    tree.append(&mut trees[7]);

    tree
}

pub fn chunk_to_tree(chunk: [[[usize; CHUNK_SIZE]; CHUNK_SIZE]; CHUNK_SIZE]) -> Octree {
    struct ST {
        parent: usize,
        idx: Vector3<f32>,
        pos: Vector3<f32>,
        scale: i32,
    }

    let mut tree = vec![Node {
        leaf: [false; 8],
        pointer: [0; 8],
    }];
    let levels = (CHUNK_SIZE as f32).log2() as i32;

    let mut stack: Vec<ST> = vec![];

    let mut i = 0;
    for j in 0.. {
        let (pos, root, idx, parent, scale) = if j == 0 {
            (
                vec3(
                    CHUNK_SIZE as f32 / 2.0,
                    CHUNK_SIZE as f32 / 2.0,
                    CHUNK_SIZE as f32 / 2.0,
                ),
                true,
                vec3(0.0, 0.0, 0.0),
                0,
                1,
            )
        } else if !stack.is_empty() {
            let s = stack.pop().unwrap();
            (s.pos, false, s.idx, s.parent, s.scale)
        } else {
            break;
        };

        let mut v = Node {
            leaf: [true; 8],
            pointer: [0; 8],
        };
        let size = 2.0_f32.powf((levels - scale) as f32);

        let mut b = false;
        for j in 0..8 {
            let jdx = Node::position(j);
            let np = pos + jdx * size * 0.5;

            if scale >= levels {
                v.leaf[j] = true;
                v.pointer[j] = chunk[np.x as usize][np.y as usize][np.z as usize];
                b = b || v.pointer[j] != 0;
            } else {
                b = true;
                stack.push(ST {
                    parent: i,
                    idx: jdx,
                    pos: np,
                    scale: scale + 1,
                });
            }
        }

        if b && !root {
            let uidx = Node::idx(idx);
            tree[parent].leaf[uidx] = false;
            tree[parent].pointer[uidx] = i;
        }
        if b {
            tree.push(v);
            i += 1;
        }
    }

    // let v = tree[0].clone();
    // let v = collapse(&mut tree, v);
    // tree[0] = v;
    tree
}

pub fn tree_to_chunk(tree: Octree) -> [[[usize; CHUNK_SIZE]; CHUNK_SIZE]; CHUNK_SIZE] {
    let mut c = [[[0; CHUNK_SIZE]; CHUNK_SIZE]; CHUNK_SIZE];
    let rad = (CHUNK_SIZE as f32) * 0.5;
    for (x, row_x) in c.iter_mut().enumerate() {
        for (y, row_y) in row_x.iter_mut().enumerate() {
            for (z, b) in row_y.iter_mut().enumerate() {
                *b = octree_get(
                    &tree,
                    CHUNK_SIZE as f32,
                    &vec3((x as f32) - rad, (y as f32) - rad, (z as f32) - rad),
                );
            }
        }
    }
    c
}

// -----------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_convert() {
        let mut chunk = [[[0; CHUNK_SIZE]; CHUNK_SIZE]; CHUNK_SIZE];
        chunk[12][4][2] = 1;
        chunk[3][6][11] = 1;
        chunk[9][9][9] = 1;
        chunk[0][0][1] = 1;
        let tree = chunk_to_tree(chunk);
        print!("{:?}", tree);
        println!("{}",tree.len());
        assert_eq!(
            1,
            octree_get(
                &tree,
                CHUNK_SIZE as f32,
                &(vec3(12.0, 4.0, 2.0) + 2.0 - 0.5 * CHUNK_SIZE as f32)
            )
        );
        assert_eq!(chunk, tree_to_chunk(tree));
    }
}
