extern crate rayon;

use na::{Isometry3, Translation3, UnitQuaternion, Vector3};
use nalgebra_glm::Vec3;

use super::Vertex;
use super::VertexBuffer;
use std::sync::Arc;

use vulkano::sync::GpuFuture;

const CHUNK_SIZE: usize = 32;

use super::RaycastCuboid;

#[rustfmt::skip]
pub const CUBE_CORNERS: [CubeCorner; 8] = [
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
pub const CUBE_FACES: [Face; 6] = [
    Face { indices: [0, 1, 3, 3, 1, 2], facing: Offset { right:  0, up:  0, front: -1 }, normal: ( 0.0,  0.0, -1.0) },
    Face { indices: [1, 5, 2, 2, 5, 6], facing: Offset { right:  1, up:  0, front:  0 }, normal: ( 1.0,  0.0,  1.0) },
    Face { indices: [5, 4, 6, 6, 4, 7], facing: Offset { right:  0, up:  0, front:  1 }, normal: ( 0.0,  0.0,  1.0) },
    Face { indices: [4, 0, 7, 7, 0, 3], facing: Offset { right: -1, up:  0, front:  0 }, normal: (-1.0,  0.0,  0.0) },
    Face { indices: [3, 2, 7, 7, 2, 6], facing: Offset { right:  0, up:  1, front:  0 }, normal: ( 0.0,  1.0,  1.0) },
    Face { indices: [4, 5, 0, 0, 5, 1], facing: Offset { right:  0, up: -1, front:  0 }, normal: ( 0.0, -1.0,  0.0) },
];

pub struct Chunk {
    pub cells: Vec<u8>,
    vbuf: VertexBuffer,
    positions: Vec<(f32, f32, f32)>,
    nearby_cuboids_offsets: Vec<(i32, i32, i32)>,
}

pub struct CubeCorner {
    pub position: (f32, f32, f32),
    neighbors: [Offset; 8],
}

pub struct Face {
    pub indices: [usize; 6],
    facing: Offset,
    pub normal: (f32, f32, f32),
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
                    .map(|_| (0..CHUNK_SIZE).map(|_| rand::random()))
            })
            .flatten()
            .flatten()
            .collect();

        let nearby_cuboids_offsets = Self::generate_nearby_cuboids_offsets();

        Chunk {
            cells: cells,
            vbuf: make_empty_vbuf(queue),
            positions: vec![],
            nearby_cuboids_offsets,
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

    pub fn randomize_state(&mut self) {
        // doesn't actually randomize it :p

        // self.cells = (0..CHUNK_SIZE)
        //     .map(|_| {
        //         (0..CHUNK_SIZE)
        //             .map(|_| (0..CHUNK_SIZE).map(|_| rand::random()).collect::<Vec<_>>())
        //             .collect::<Vec<_>>()
        //     })
        //     .flatten()
        //     .flatten()
        //     .collect();

        let s = CHUNK_SIZE;
        let coef = 0.5;
        self.cells = (0..s)
            .map(move |z| {
                (0..s).map(move |y| {
                    (0..s).map(move |x| {
                        if (x as f32 * coef).sin()
                            + ((y * 2) as f32 * coef).sin()
                            + (z as f32 * coef * 0.5).sin()
                            > 0.0
                        {
                            1
                        } else {
                            0
                        }
                    })
                })
            })
            .flatten()
            .flatten()
            .collect::<Vec<_>>();
    }

    pub fn generate_cuboids_close_to(&self, camera_position: Vec3) -> Vec<RaycastCuboid> {
        // generates a list of not-air cuboids for testing ray intersections with.
        self.nearby_cuboids_offsets
            .iter()
            .filter_map(|(x_off, y_off, z_off)| {
                // double conversion is to round down...
                let new_x = ((camera_position.x + (*x_off as f32)) as i32) as f32;
                let new_y = ((camera_position.z + (*z_off as f32)) as i32) as f32;
                let new_z = ((camera_position.y + (*y_off as f32)) as i32) as f32;

                let out_of_bounds = (new_x < 0.0)
                    || (new_y < 0.0)
                    || (new_z) < 0.0
                    || (new_x >= (CHUNK_SIZE as f32))
                    || (new_y >= (CHUNK_SIZE as f32))
                    || (new_z >= (CHUNK_SIZE as f32));
                if !out_of_bounds {
                    let idx = xyz_to_linear(new_x as usize, new_y as usize, new_z as usize);
                    if self.cells[idx] > 0 {
                        // finally, the interesting part: we found a block close to the camera!
                        // generate a cuboid for it
                        let isometry = Isometry3::from_parts(
                            Translation3::new(new_x, new_z, new_y),
                            UnitQuaternion::from_scaled_axis(Vector3::y() * 0.0),
                        );

                        Some((isometry, idx))
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
    }

    fn generate_vertices(&self) -> Vec<Vertex> {
        let min_idx = (CHUNK_SIZE * CHUNK_SIZE) + CHUNK_SIZE + 1;
        let max_idx = (CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE) - min_idx;
        (min_idx..max_idx)
            .filter_map(|idx| {
                if self.cells[idx] > 0 {
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
        if self.cells[idx] > 0 && self.count_neighbors(idx) < 26 {
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
                            let x = (idx % CHUNK_SIZE) as f32 / (CHUNK_SIZE as f32);
                            let y = (idx / CHUNK_SIZE % CHUNK_SIZE) as f32 / (CHUNK_SIZE as f32);
                            let z = (idx / (CHUNK_SIZE * CHUNK_SIZE)) as f32 / (CHUNK_SIZE as f32);
                            let color = if self.cells[idx] == 2 {
                                (value, 0.0, 0.0, 1.0)
                            } else {
                                (value * x, value * y, value * z, 1.0)
                            };

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

    fn get_value_of_vertex(&self, base_idx: usize, offsets: &[Offset]) -> f32 {
        let mut neighbor_count = 0;

        for offset in offsets.iter() {
            let idx_offset = offset.get_idx_offset();
            // pray it doesn't overflow in the usize -> i32 conversion
            // would happen with size > 1200, which is pretty extreme but possible.
            // usize overflows at size > 1500, which isn't so great either.
            // turns out 3d is hard! ;)
            let new_idx = ((base_idx as i32) + idx_offset) as usize;
            if self.cells[new_idx] > 0 {
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

        neighbors.iter().filter(|&x| *x > 0).count()
    }

    fn generate_nearby_cuboids_offsets() -> Vec<(i32, i32, i32)> {
        let max_dist = 9;
        let mut offsets = vec![(0, 0, 0)];

        // the hard part is making the cuboids in an order such that the closest comes first.
        for distance in 1..=max_dist {
            for offset_1 in -max_dist..=max_dist {
                for offset_2 in -max_dist..=max_dist {
                    offsets.push((distance, offset_1, offset_2));
                    offsets.push((offset_2, distance, offset_1));
                    offsets.push((offset_1, offset_2, distance));
                    offsets.push((-distance, offset_1, offset_2));
                    offsets.push((offset_2, -distance, offset_1));
                    offsets.push((offset_1, offset_2, -distance));
                }
            }
        }

        // remove duplicates
        let mut no_duplicates = vec![];
        for offset in offsets.iter() {
            if !no_duplicates.contains(&offset) {
                no_duplicates.push(offset);
            }
        }

        println!("Offsets len: {}", no_duplicates.len());
        println!("Expected len: {}", 19 * 19 * 19);

        offsets
    }
}

pub fn vbuf_from_verts(queue: Arc<vulkano::device::Queue>, vertices: Vec<Vertex>) -> VertexBuffer {
    let (buffer, future) = vulkano::buffer::immutable::ImmutableBuffer::from_iter(
        vertices.iter().cloned(),
        vulkano::buffer::BufferUsage::vertex_buffer(),
        queue.clone(),
    )
    .unwrap();
    future.flush().unwrap();

    buffer
}

pub fn make_empty_vbuf(queue: Arc<vulkano::device::Queue>) -> VertexBuffer {
    vbuf_from_verts(queue, vec![])
}

pub fn xyz_to_linear(x: usize, y: usize, z: usize) -> usize {
    z * (CHUNK_SIZE * CHUNK_SIZE) + y * CHUNK_SIZE + x
}

impl Offset {
    fn get_idx_offset(&self) -> i32 {
        let sz = CHUNK_SIZE as i32;

        self.up * (sz * sz) + self.front * sz + self.right
    }
}

impl Face {
    fn is_visible(&self, cells: &[u8], idx: usize) -> bool {
        let neighbor_idx = ((idx as i32) + self.facing.get_idx_offset()) as usize;

        cells[neighbor_idx] == 0
    }
}
