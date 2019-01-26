extern crate rayon;
use rayon::prelude::*;
use std::collections::vec_deque::VecDeque;

const MAX_CACHE_VBUFS: usize = 1024;

#[derive(Copy, Clone, Debug)]
pub struct Vertex {
    pub position: (f32, f32, f32),
    pub color: (f32, f32, f32, f32),
    pub normal: (f32, f32, f32),
}

type VertexBuffer = std::sync::Arc<vulkano::buffer::cpu_access::CpuAccessibleBuffer<[Vertex]>>;

#[rustfmt::skip]
const CUBE_CORNERS: [CubeCorner; 8] = [
    CubeCorner { position: (-0.5, -0.5, -0.5), neighbors: [Offset { right: -1, up: -1, front: -1},   Offset { right: -1, up: -1, front:  0},   Offset { right: -1, up:  0, front: -1},   Offset { right: -1, up:  0, front:  0},  Offset { right:  0, up: -1, front: -1},    Offset { right:  0, up: -1, front:  0},   Offset { right:  0, up:  0, front: -1},  Offset { right:  0, up:  0, front:  0} ] },
    CubeCorner { position: ( 0.5, -0.5, -0.5), neighbors: [Offset { right:  0, up: -1, front: -1},   Offset { right:  0, up: -1, front:  0},   Offset { right:  0, up:  0, front: -1},   Offset { right:  0, up:  0, front:  0},  Offset { right:  1, up: -1, front: -1},    Offset { right:  1, up: -1, front:  0},   Offset { right:  1, up:  0, front: -1},  Offset { right:  1, up:  0, front:  0} ] },
    CubeCorner { position: ( 0.5,  0.5, -0.5), neighbors: [Offset { right:  0, up:  0, front: -1},   Offset { right:  0, up:  0, front:  0},   Offset { right:  0, up:  1, front: -1},   Offset { right:  0, up:  1, front:  0},  Offset { right:  1, up:  0, front: -1},    Offset { right:  1, up:  0, front:  0},   Offset { right:  1, up:  1, front: -1},  Offset { right:  1, up:  1, front:  0} ] },
    CubeCorner { position: (-0.5,  0.5, -0.5), neighbors: [Offset { right: -1, up:  0, front: -1},   Offset { right: -1, up:  0, front:  0},   Offset { right: -1, up:  1, front: -1},   Offset { right: -1, up:  1, front:  0},  Offset { right:  0, up:  0, front: -1},    Offset { right:  0, up:  0, front:  0},   Offset { right:  0, up:  1, front: -1},  Offset { right:  0, up:  1, front:  0} ] },
    CubeCorner { position: (-0.5, -0.5,  0.5), neighbors: [Offset { right: -1, up: -1, front:  0},   Offset { right: -1, up: -1, front:  1},   Offset { right: -1, up:  0, front:  0},   Offset { right: -1, up:  0, front:  1},  Offset { right:  0, up: -1, front:  0},    Offset { right:  0, up: -1, front:  1},   Offset { right:  0, up:  0, front:  0},  Offset { right:  0, up:  0, front:  1} ] },
    CubeCorner { position: ( 0.5, -0.5,  0.5), neighbors: [Offset { right:  0, up: -1, front:  0},   Offset { right:  0, up: -1, front:  1},   Offset { right:  0, up:  0, front:  0},   Offset { right:  0, up:  0, front:  1},  Offset { right:  1, up: -1, front:  0},    Offset { right:  1, up: -1, front:  1},   Offset { right:  1, up:  0, front:  0},  Offset { right:  1, up:  0, front:  1} ] },
    CubeCorner { position: ( 0.5,  0.5,  0.5), neighbors: [Offset { right:  0, up:  0, front:  0},   Offset { right:  0, up:  0, front:  1},   Offset { right:  0, up:  1, front:  0},   Offset { right:  0, up:  1, front:  1},  Offset { right:  1, up:  0, front:  0},    Offset { right:  1, up:  0, front:  1},   Offset { right:  1, up:  1, front:  0},  Offset { right:  1, up:  1, front:  1} ] },
    CubeCorner { position: (-0.5,  0.5,  0.5), neighbors: [Offset { right: -1, up:  0, front:  0},   Offset { right: -1, up:  0, front:  1},   Offset { right: -1, up:  1, front:  0},   Offset { right: -1, up:  1, front:  1},  Offset { right:  0, up:  0, front:  0},    Offset { right:  0, up:  0, front:  1},   Offset { right:  0, up:  1, front:  0},  Offset { right:  0, up:  1, front:  1} ] },
];

#[rustfmt::skip]
const CUBE_FACES: [Face; 6] = [
    Face { indices: [0, 1, 3, 3, 1, 2], facing: Offset { right:  0, up:  0, front: -1 }, normal: ( 0.0,  0.0, -1.0) },
    Face { indices: [1, 5, 2, 2, 5, 6], facing: Offset { right:  1, up:  0, front:  0 }, normal: ( 1.0,  0.0,  1.0) },
    Face { indices: [5, 4, 6, 6, 4, 7], facing: Offset { right:  0, up:  0, front:  1 }, normal: ( 0.0,  0.0,  1.0) },
    Face { indices: [4, 0, 7, 7, 0, 3], facing: Offset { right: -1, up:  0, front:  0 }, normal: (-1.0,  0.0,  0.0) },
    Face { indices: [3, 2, 7, 7, 2, 6], facing: Offset { right:  0, up:  1, front:  0 }, normal: ( 0.0,  1.0,  1.0) },
    Face { indices: [4, 5, 0, 0, 5, 1], facing: Offset { right:  0, up: -1, front:  0 }, normal: ( 0.0, -1.0,  0.0) },
];

struct CubeCorner {
    position: (f32, f32, f32),
    neighbors: [Offset; 8],
}

struct Face {
    indices: [usize; 6],
    facing: Offset,
    normal: (f32, f32, f32),
}

struct Offset {
    up: i32,
    right: i32,
    front: i32,
}

pub struct VbufCache {
    sector_vertices: Vec<Vec<Vertex>>,
    pub vertex_buffers: Vec<Option<VertexBuffer>>,
    positions: Vec<(f32, f32, f32)>,
    chunked_indices: Vec<Vec<usize>>,
    cached_indices: VecDeque<usize>,
}

use super::ca;
use super::SIZE;

impl VbufCache {
    pub fn new() -> VbufCache {
        println!("VbufCache initialized...");
        let positions = generate_positions();
        println!("Done generating positions.");
        let chunked_indices = generate_chunked_indices();
        println!("Done generating indices.");

        let num_sectors = super::SIZE * super::SIZE * super::SIZE;
        let sector_vertices = (0..num_sectors).map(|_| vec![]).collect();

        VbufCache {
            sector_vertices,
            vertex_buffers: vec![None; num_sectors],
            positions,
            chunked_indices,
            cached_indices: VecDeque::new(),
        }
    }

    pub fn get_vbuf_at_idx(
        &mut self,
        idx: usize,
        cells: &[u8],
        device: &std::sync::Arc<vulkano::device::Device>,
    ) -> VertexBuffer {
        // the thing it returns is already cloned, don't worry :)

        // maybe move this somewhere less often called? :P
        self.uncache_oldest();

        // had to do the clone first to satisfy the borrow checker,
        // hopefully that doesn't cause problems :/
        match self.vertex_buffers[idx].clone() {
            Some(vbuf) => vbuf,
            None => {
                self.vertex_buffers[idx] = Some(self.generate_vbuf_for_idx(idx, device, cells));
                self.vertex_buffers[idx].clone().unwrap()
            }
        }
    }

    pub fn get_num_cached_vbufs(&self) -> usize {
        self.cached_indices.len()
    }

    fn generate_vbuf_for_idx(
        &mut self,
        idx: usize,
        device: &std::sync::Arc<vulkano::device::Device>,
        cells: &[u8],
    ) -> VertexBuffer {
        if self.sector_vertices[idx].is_empty() {
            self.update_vertices_at_idx(idx, cells);
        }
        let vertices = &self.sector_vertices[idx];
        self.cached_indices.push_back(idx);

        vulkano::buffer::cpu_access::CpuAccessibleBuffer::from_iter(
            device.clone(),
            vulkano::buffer::BufferUsage::vertex_buffer(),
            vertices.iter().cloned(),
        )
        .expect("failed to create buffer")
    }

    fn update_vertices_at_idx(&mut self, idx: usize, cells: &[u8]) {
        let indices = &self.chunked_indices[idx];
        self.sector_vertices[idx] = generate_vertices_for_indices(cells, &self.positions, &indices);
    }

    fn uncache_oldest(&mut self) {
        if self.cached_indices.len() > MAX_CACHE_VBUFS {
            let how_much_over = self.cached_indices.len() - MAX_CACHE_VBUFS;
            let indices = self.cached_indices.drain(0..how_much_over);

            for idx in indices {
                self.vertex_buffers[idx] = None;
                self.sector_vertices[idx] = vec![];
            }
        }
    }
}

fn generate_vertices_for_indices(
    cells: &[u8],
    positions: &[(f32, f32, f32)],
    indices: &[usize],
) -> Vec<Vertex> {
    indices
        .par_iter()
        .map(|&idx| generate_verts_for_cube(cells, idx, positions[idx]))
        .flatten()
        .collect()
}

fn generate_verts_for_cube(cells: &[u8], idx: usize, offset: (f32, f32, f32)) -> Vec<Vertex> {
    // make sure cell is alive and not totally obscured
    let choice = rand::random::<i32>() % 3;
    if cells[idx] > 0 && ca::count_neighbors(cells, idx) < 26 {
        CUBE_FACES
            .iter()
            .filter_map(move |face| {
                if face.is_visible(cells, idx) {
                    Some(face.indices.iter().map(move |&v_idx| {
                        let corner = &CUBE_CORNERS[v_idx];
                        let pos = corner.position;
                        let offsets = &corner.neighbors;

                        // determine ao of vertex
                        let value = get_value_of_vertex(cells, idx, offsets);
                        let color;
                        if choice == 0 {
                            color = (value, value * 0.8, value * 0.8, 1.0);
                        } else if choice == 1 {
                            color = (value * 0.8, value, value * 0.8, 1.0);
                        } else {
                            color = (value * 0.8, value * 0.8, value, 1.0);
                        }

                        Vertex {
                            position: (pos.0 + offset.0, pos.1 + offset.1, pos.2 + offset.2),
                            color,
                            normal: face.normal,
                        }
                    }))
                } else {
                    None
                }
            })
            .flatten()
            .collect()
    } else {
        Vec::new()
    }
}

fn get_value_of_vertex(cells: &[u8], base_idx: usize, offsets: &[Offset]) -> f32 {
    let mut neighbor_count = 0;

    for offset in offsets.iter() {
        let idx_offset = offset.get_idx_offset();
        let new_idx = ((base_idx as i32) + idx_offset) as usize;
        if cells[new_idx] > 0 {
            neighbor_count += 1;
        }
    }

    1.0 - (neighbor_count as f32 / 13.0)
}

fn generate_chunked_indices() -> Vec<Vec<usize>> {
    use super::SECTOR_SIDE_LEN;
    use super::SIZE;
    let world_size_chunks = SIZE / SECTOR_SIDE_LEN;
    println!("wsc: {}", world_size_chunks);

    // todo: turn this into a ridiculously far-right iter chain
    let mut chunked_indices = Vec::new();
    for base_z_idx in 0..world_size_chunks {
        let base_z = base_z_idx * SECTOR_SIDE_LEN;

        for base_y_idx in 0..world_size_chunks {
            let base_y = base_y_idx * SECTOR_SIDE_LEN;

            for base_x_idx in 0..world_size_chunks {
                let base_x = base_x_idx * SECTOR_SIDE_LEN;

                let mut indices = Vec::new();
                for sub_z in 0..SECTOR_SIDE_LEN {
                    for sub_y in 0..SECTOR_SIDE_LEN {
                        for sub_x in 0..SECTOR_SIDE_LEN {
                            indices.push(xyz_to_linear(
                                base_x + sub_x,
                                base_y + sub_y,
                                base_z + sub_z,
                            ));
                        }
                    }
                }

                chunked_indices.push(indices);
            }
        }
    }

    chunked_indices
}

fn xyz_to_linear(x: usize, y: usize, z: usize) -> usize {
    // uses [z][y][x]
    use super::SIZE;
    z * SIZE * SIZE + y * SIZE + x
}

fn generate_positions() -> Vec<(f32, f32, f32)> {
    (0..SIZE)
        .map(|y| {
            (0..SIZE)
                .map(|z| {
                    (0..SIZE)
                        .map(|x| (x as f32, y as f32, z as f32))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        })
        .flatten()
        .flatten()
        .collect()
}

impl Offset {
    fn get_idx_offset(&self) -> i32 {
        let sz = SIZE as i32;

        self.up * (sz * sz) + self.front * sz + self.right
    }
}

impl Face {
    fn is_visible(&self, cells: &[u8], idx: usize) -> bool {
        let neighbor_idx = ((idx as i32) + self.facing.get_idx_offset()) as usize;

        cells[neighbor_idx] == 0
    }
}
