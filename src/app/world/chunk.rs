extern crate rayon;

use na::{Isometry3, Translation3, UnitQuaternion, Vector3};
use nalgebra_glm::Vec3;

use super::super::super::utils::*;
use std::sync::Arc;

extern crate noise;
use noise::{Perlin, NoiseFn};

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

    // coordinates of this chunk
    pub chunk_coord: ChunkCoordinate,

    pub has_been_modified: bool,
    // coordinates of the corner of this chunk in 3d space
    offset: (f32, f32, f32),

    // the vertex buffer is cached here
    vbuf: VertexBuffer,

    // a list of the 3d coordinates, listed in the same order as cells
    positions: Vec<(f32, f32, f32)>,

    // 3d offsets for which cells are considered "nearby"
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
    pub fn new(
        queue: Arc<vulkano::device::Queue>,
        chunk_coord: ChunkCoordinate,
        offset: (f32, f32, f32),
    ) -> Self {
        let cells = (0..CHUNK_SIZE)
            .map(|_| (0..CHUNK_SIZE).map(|_| (0..CHUNK_SIZE).map(|_| rand::random())))
            .flatten()
            .flatten()
            .collect();

        let nearby_cuboids_offsets = Self::generate_nearby_cuboids_offsets();

        // we need to adjust offset so that the chunk is centered around the offset
        // otherwise, the chunk's top-left-front corner will be on the offset
        // and not the center
        let half = (CHUNK_SIZE / 2) as f32;
        let offset = (offset.0 - half, offset.1 - half, offset.2 - half);

        Chunk {
            cells,
            chunk_coord,
            has_been_modified: false,
            offset,
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
        let noise_gen = Perlin::new();

        // borrow checker complains otherwise because closures are still picky
        let ccx = self.chunk_coord.x;
        let ccy = self.chunk_coord.y;
        let ccz = self.chunk_coord.z;
        self.cells = (0..(CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE)).map(|idx| {
            // z and y need to be swapped here because the chunks are kinda twisted. gotta fix this.
            let x = idx % CHUNK_SIZE;
            let z = idx % (CHUNK_SIZE * CHUNK_SIZE) / CHUNK_SIZE;
            let y = idx / (CHUNK_SIZE * CHUNK_SIZE);
            (noise_gen.get([((x as f64) / (CHUNK_SIZE as f64) + (ccx as f64)) / 1.5, ((y as f64) / (CHUNK_SIZE as f64) + (ccy as f64)) / 1.5, ((z as f64) / (CHUNK_SIZE as f64) + (ccz as f64)) / 1.5]) * 0.8).round() as u8
        }).collect::<Vec<_>>();

        self.has_been_modified = true;
    }

    pub fn generate_cuboids_close_to(&self, camera_position: Vec3) -> Vec<CuboidOffset> {
        // generates a list of not-air cuboids for testing ray intersections with.
        // first, subtract the out chunk offset from the camera position so that the indices are in the same range
        // basically, we get around the problem of the chunks having offsets by pretending that all chunks are in the center
        // and that the camera is somewhere in the center chunk.

        // these camera positions are relative to 0 0 0 in our chunk,
        // so even if the chunk is faaaaar offset from the center,
        // the camera could be in the same area and therefore have coordinates within bounds.
        let relative_cam_x = camera_position.x - self.offset.0;
        let relative_cam_y = camera_position.y - self.offset.1;
        let relative_cam_z = camera_position.z - self.offset.2;

        // now check whether the camera is nearby any filled cells in the current chunk.
        // to do this, we iterate through the pre-generated list nearby_cuboids_offsets
        // and offset the rounded camera position by every position in there, then check
        // that position to see if it is filled.
        // if it is, we generate a CuboidOffset for it which is later returned.
        self.nearby_cuboids_offsets
            .iter()
            .filter_map(|(x_off, y_off, z_off)| {
                // double conversion is to round down...
                let new_rel_x = ((relative_cam_x + (*x_off as f32)) as i32) as f32;
                let new_rel_y = ((relative_cam_y + (*y_off as f32)) as i32) as f32;
                let new_rel_z = ((relative_cam_z + (*z_off as f32)) as i32) as f32;

                let out_of_bounds = (new_rel_x < 0.0)
                    || (new_rel_y < 0.0)
                    || (new_rel_z) < 0.0
                    || (new_rel_x >= (CHUNK_SIZE as f32))
                    || (new_rel_y >= (CHUNK_SIZE as f32))
                    || (new_rel_z >= (CHUNK_SIZE as f32));
                if !out_of_bounds {
                    let idx =
                        xyz_to_linear(new_rel_x as usize, new_rel_y as usize, new_rel_z as usize);
                    if self.cells[idx] > 0 {
                        // finally, the interesting part: we found a block close to the camera!
                        // generate a cuboid for it
                        let new_absolute_x = new_rel_x + self.offset.0;
                        let new_absolute_y = new_rel_y + self.offset.1;
                        let new_absolute_z = new_rel_z + self.offset.2;

                        let isometry = Isometry3::from_parts(
                            Translation3::new(new_absolute_x, new_absolute_y, new_absolute_z),
                            UnitQuaternion::from_scaled_axis(Vector3::y() * 0.0),
                        );

                        Some(isometry)
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
        let total_cells = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;
        (0..total_cells)
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
        // generates vertices for the cube at idx

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
                            let x = (offset.0 as f32) / (CHUNK_SIZE as f32);
                            let y = (offset.1 as f32) / (CHUNK_SIZE as f32);
                            let z = (offset.2 as f32) / (CHUNK_SIZE as f32);
                            let color = if self.cells[idx] == 2 {
                                (value, 0.0, 0.0, 1.0)
                            } else {
                                (value * x, value * y, value * z, 1.0)
                            };

                            Vertex {
                                position: (
                                    pos.0 + offset.0 + self.offset.0,
                                    pos.1 + offset.1 + self.offset.1,
                                    pos.2 + offset.2 + self.offset.2,
                                ),
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
            let new_idx = ((base_idx as i32) + idx_offset) as usize;

            // first comparison prevents overflow
            if new_idx < (CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE) && self.cells[new_idx] > 0 {
                neighbor_count += 1;
            }
        }

        1.0 - (neighbor_count as f32 / 13.0)
    }

    fn count_neighbors(&self, idx: usize) -> usize {
        let size = CHUNK_SIZE;
        let min_idx = (size * size) + size + 1;
        let max_idx = (size * size * size) - min_idx;

        // if the index is in this range we don't have to worry about going OOB
        let neighbors = if idx > min_idx && idx < max_idx {
            [
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
            ]
        } else {
            // otherwise just don't count neighbors
            [0; 26]
        };

        neighbors.iter().filter(|&x| *x > 0).count()
    }

    fn generate_nearby_cuboids_offsets() -> Vec<(i32, i32, i32)> {
        let max_dist = 9;
        let mut offsets = vec![(0, 0, 0)];

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

        offsets
    }
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
        let x = neighbor_idx % CHUNK_SIZE;
        let y = neighbor_idx % (CHUNK_SIZE * CHUNK_SIZE) / CHUNK_SIZE;
        let z = neighbor_idx / (CHUNK_SIZE * CHUNK_SIZE);

        if x < 2
            || x >= CHUNK_SIZE - 2
            || y < 2
            || y >= CHUNK_SIZE - 2
            || z < 2
            || z >= CHUNK_SIZE - 2
        {
            // just assume it's visible if it's close to the edge of the chunk,
            // because we can't tell without looking at other chunks
            true
        } else {
            cells[neighbor_idx] == 0
        }
    }
}
