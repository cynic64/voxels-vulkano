extern crate rayon;
use rayon::prelude::*;

use super::Vertex;
use super::VertexBuffer;
use std::sync::Arc;

use rand::Rng;
use vulkano::sync::GpuFuture;

const CHUNK_SIZE: usize = 32;

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

pub struct Chunk {
    cells: Vec<bool>,
    vbuf: VertexBuffer,
    positions: Vec<(f32, f32, f32)>,
}

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

impl Chunk {
    pub fn new(queue: Arc<vulkano::device::Queue>) -> Self {
        let cells = (0..CHUNK_SIZE)
            .map(|_| {
                (0..CHUNK_SIZE)
                    .map(|_| (0..CHUNK_SIZE).map(|_| rand::random()).collect::<Vec<_>>())
                    .collect::<Vec<_>>()
            })
            .flatten()
            .flatten()
            .collect();

        Chunk {
            cells: cells,
            vbuf: make_empty_vbuf(queue),
            positions: vec![],
        }
    }

    pub fn update_vbuf(&mut self, queue: Arc<vulkano::device::Queue>) {
        let vertices = self.generate_vertices();

        self.vbuf = vbuf_from_verts(queue, vertices);
    }

    pub fn update_positions(&mut self) {
        self.positions = (0..CHUNK_SIZE)
            .map(|y| {
                (0..CHUNK_SIZE)
                    .map(|z| {
                        (0..CHUNK_SIZE)
                            .map(|x| (x as f32, y as f32, z as f32))
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .flatten()
            .flatten()
            .collect();
    }

    pub fn get_vbuf(&self) -> VertexBuffer {
        self.vbuf.clone()
    }

    fn generate_vertices(&self) -> Vec<Vertex> {
        let min_idx = (CHUNK_SIZE * CHUNK_SIZE) + CHUNK_SIZE + 1;
        let max_idx = (CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE) - min_idx;
        (min_idx..max_idx)
            .filter_map(|idx| {
                if self.cells[idx] {
                    Some(self.generate_verts_for_cube(idx))
                } else {
                    None
                }
            })
            .flatten()
            .collect()
    }

    fn generate_verts_for_cube(&self, idx: usize) -> Vec<Vertex> {
        // make sure cell is alive and not totally obscured
        let offset = self.positions[idx];
        if self.cells[idx] && self.count_neighbors(idx) < 26 {
            CUBE_FACES
                .iter()
                .filter_map(move |face| {
                    if face.is_visible(&self.cells, idx) {
                        Some(face.indices.iter().map(move |&v_idx| {
                            let corner = &CUBE_CORNERS[v_idx];
                            let pos = corner.position;

                            // determine ao of vertex
                            let offsets = &corner.neighbors;
                            let value = self.get_value_of_vertex(idx, offsets);

                            Vertex {
                                position: (pos.0 + offset.0, pos.1 + offset.1, pos.2 + offset.2),
                                color: (value, value, value, 1.0),
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

    fn get_value_of_vertex(&self, base_idx: usize, offsets: &[Offset]) -> f32 {
        let mut neighbor_count = 0;

        for offset in offsets.iter() {
            let idx_offset = offset.get_idx_offset();
            // pray it doesn't overflow in the usize -> i32 conversion
            // would happen with size > 1200, which is pretty extreme but possible.
            // usize overflows at size > 1500, which isn't so great either.
            // turns out 3d is hard! ;)
            let new_idx = ((base_idx as i32) + idx_offset) as usize;
            if self.cells[new_idx] {
                neighbor_count += 1;
            }
        }

        1.0 - (neighbor_count as f32 / 13.0)
    }

    fn count_neighbors(&self, idx: usize) -> usize {
        let size = CHUNK_SIZE;

        let neighbors = [
            self.cells[idx + (size * size) + size + 1],
            self.cells[idx + (size * size) + size],
            self.cells[idx + (size * size) + size - 1],
            self.cells[idx + (size * size) + 1],
            self.cells[idx + (size * size)],
            self.cells[idx + (size * size) - 1],
            self.cells[idx + (size * size) - size + 1],
            self.cells[idx + (size * size) - size],
            self.cells[idx + (size * size) - size - 1],
            self.cells[idx + size + 1],
            self.cells[idx + size],
            self.cells[idx + size - 1],
            self.cells[idx + 1],
            self.cells[idx - 1],
            self.cells[idx - size + 1],
            self.cells[idx - size],
            self.cells[idx - size - 1],
            self.cells[idx - (size * size) + size + 1],
            self.cells[idx - (size * size) + size],
            self.cells[idx - (size * size) + size - 1],
            self.cells[idx - (size * size) + 1],
            self.cells[idx - (size * size)],
            self.cells[idx - (size * size) - 1],
            self.cells[idx - (size * size) - size + 1],
            self.cells[idx - (size * size) - size],
            self.cells[idx - (size * size) - size - 1],
        ];

        neighbors.iter().filter(|&x| *x).count()
    }
}

fn vbuf_from_verts(queue: Arc<vulkano::device::Queue>, vertices: Vec<Vertex>) -> VertexBuffer {
    let (buffer, future) = vulkano::buffer::immutable::ImmutableBuffer::from_iter(
        vertices.iter().cloned(),
        vulkano::buffer::BufferUsage::vertex_buffer(),
        queue.clone(),
    )
    .unwrap();
    future.flush().unwrap();

    buffer
}

fn make_empty_vbuf(queue: Arc<vulkano::device::Queue>) -> VertexBuffer {
    vbuf_from_verts(queue, vec![])
}

impl Offset {
    fn get_idx_offset(&self) -> i32 {
        let sz = CHUNK_SIZE as i32;

        self.up * (sz * sz) + self.front * sz + self.right
    }
}

impl Face {
    fn is_visible(&self, cells: &[bool], idx: usize) -> bool {
        let neighbor_idx = ((idx as i32) + self.facing.get_idx_offset()) as usize;

        !cells[neighbor_idx]
    }
}
