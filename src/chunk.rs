use super::glm::*;
use super::grid::*;

pub struct Chunks {
    chunks: Vec<Chunk>,
    map: Vec<Vec<Vec<usize>>>,
}
pub struct ChunksU {
    pub chunks: Vec<Vec<Vec<(u8,u8,u8)>>>,
    pub blocks: Vec<Vec<Vec<Block>>>,
}

impl ChunksU {
    fn new() -> Self {
        ChunksU {
            chunks: Vec::new(),
            blocks: Vec::new(),
        }
    }
}

pub fn gen_chunks() -> Chunks {
    let mut c = Chunks::new();
    let mut i = 0;
    for (x, row_x) in c.map.iter_mut().enumerate() {
        for (y, row_y) in row_x.iter_mut().enumerate() {
            for (z, n) in row_y.iter_mut().enumerate() {
                c.chunks[i] = gen_chunk(ivec3(x as i32,y as i32,z as i32));
                *n = i;
                i += 1;
            }
        }
    };
    c
}

impl Chunks {
    fn new() -> Self {
        Chunks {
            chunks: vec![[[[0; CHUNK_SIZE]; CHUNK_SIZE]; CHUNK_SIZE]; CHUNK_NUM*CHUNK_NUM*CHUNK_NUM],
            map: vec![vec![vec![0; CHUNK_NUM]; CHUNK_NUM]; CHUNK_NUM],
        }
    }

    fn idx_to_pos(n: usize) -> (usize, usize, usize) {
        let q = n / CHUNK_NUM;
        (CHUNK_SIZE * (q / CHUNK_NUM), CHUNK_SIZE * (q % CHUNK_NUM), CHUNK_SIZE * (n % CHUNK_NUM))
    }

    pub fn to_uniform(self) -> ChunksU {
        let mut c = ChunksU::new();
        // let s = (self.chunks.len() as f32).cbrt() as usize;
        // assert_eq!(s, CHUNK_NUM);
        // assert_eq!(self.chunks.len(), CHUNK_NUM*CHUNK_NUM*CHUNK_NUM);
        c.blocks = vec![vec![vec![0; CHUNK_SIZE*CHUNK_NUM]; CHUNK_SIZE*CHUNK_NUM]; CHUNK_SIZE*CHUNK_NUM];
        for (n,i) in self.chunks.iter().enumerate() {
            // 2 * 2 * 2
            // 4 => (0,0,1)
            // 5 => (1,0,1)
            // 6 => (0,1,1)
            // 7 => (1,1,1)
            // 2 => (0,1,0)
            // n => (n%2, (n/2)%2, (n/2)/2)

            // 0 1
            // 2 3

            // 4 5
            // 6 7
            let p = Self::idx_to_pos(n);
            for (x, row_x) in i.iter().enumerate() {
                for (y, row_y) in row_x.iter().enumerate() {
                    for (z, b) in row_y.iter().enumerate() {
                        // assert!(p.0 <= s*CHUNK_SIZE - CHUNK_SIZE, "{}", p.0);
                        c.blocks[p.0+x]
                            [p.1+y]
                            [p.2+z] = *b;
                    }
                }
            }
        }
        c.chunks = vec![vec![vec![(0,0,0); CHUNK_NUM]; CHUNK_NUM]; CHUNK_NUM];
        for (x, row_x) in self.map.iter().enumerate() {
            for (y, row_y) in row_x.iter().enumerate() {
                for (z, &n) in row_y.iter().enumerate() {
                    let p = Self::idx_to_pos(n);
                    let p = (p.0 as u8, p.1 as u8, p.2 as u8);
                    c.chunks[x][y][z] = p;
                }
            }
        }
        c
    }
}
