#[derive(Copy, Clone, Debug)]
pub struct Vertex {
    pub position: (f32, f32, f32),
    pub color: (f32, f32, f32, f32),
}

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
    Face { indices: [0, 1, 3, 3, 1, 2], facing: Offset { right: 0, up: 0, front: -1 } },
    Face { indices: [1, 5, 2, 2, 5, 6], facing: Offset { right: 1, up: 0, front: 0 } },
    Face { indices: [5, 4, 6, 6, 4, 7], facing: Offset { right: 0, up: 0, front: 1 } },
    Face { indices: [4, 0, 7, 7, 0, 3], facing: Offset { right: -1, up: 0, front: 0 } },
    Face { indices: [3, 2, 7, 7, 2, 6], facing: Offset { right: 0, up: 1, front: 0 } },
    Face { indices: [4, 5, 0, 0, 5, 1], facing: Offset { right: 0, up: -1, front: 0 } },
];

struct CubeCorner {
    position: (f32, f32, f32),
    neighbors: [Offset; 8],
}

struct Face {
    indices: [usize; 6],
    facing: Offset,
}

struct Offset {
    up: i32,
    right: i32,
    front: i32,
}

use super::SIZE;
use super::ca;

pub fn update_vbuf(
    cells: &[u8],
    positions: &[(f32, f32, f32)],
    device: &std::sync::Arc<vulkano::device::Device>,
) -> std::sync::Arc<vulkano::buffer::cpu_access::CpuAccessibleBuffer<[Vertex]>> {
    let vertices = generate_vertices_for_indices(cells, positions, &(0..1000000).collect::<Vec<_>>());
    vulkano::buffer::cpu_access::CpuAccessibleBuffer::from_iter(
        device.clone(),
        vulkano::buffer::BufferUsage::vertex_buffer(),
        vertices.iter().cloned(),
    )
    .expect("failed to create buffer")
}

fn generate_vertices(cells: &[u8], positions: &[(f32, f32, f32)]) -> Vec<Vertex> {
    positions
        .iter()
        .enumerate()
        .map(|(idx, &offset)| generate_verts_for_cube(cells, idx, offset))
        .flatten()
        .collect()
}

fn generate_vertices_for_indices(cells: &[u8], positions: &[(f32, f32, f32)], indices: &[usize]) -> Vec<Vertex> {
    indices
        .iter()
        .map(|&idx| generate_verts_for_cube(cells, idx, positions[idx]))
        .flatten()
        .collect()
}

fn generate_verts_for_cube(cells: &[u8], idx: usize, offset: (f32, f32, f32)) -> Vec<Vertex> {
    // make sure cell is alive and not totally obscured
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
                        let color = get_color_of_vertex(cells, idx, offsets);

                        Vertex {
                            position: (pos.0 + offset.0, pos.1 + offset.1, pos.2 + offset.2),
                            color,
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

fn get_color_of_vertex(cells: &[u8], base_idx: usize, offsets: &[Offset]) -> (f32, f32, f32, f32) {
    let mut neighbor_count = 0;

    for offset in offsets.iter() {
        let idx_offset = offset.get_idx_offset();
        // pray it doesn't overflow in the usize -> i32 conversion
        // would happen with size > 1200, which is pretty extreme but possible.
        // usize overflows at size > 1500, which isn't so great either.
        // turns out 3d is hard! ;)
        let new_idx = ((base_idx as i32) + idx_offset) as usize;
        if cells[new_idx] > 0 {
            neighbor_count += 1;
        }
    }

    let value = 1.0 - (neighbor_count as f32 / 26.0);

    (value, value, value, 1.0)
}

pub fn generate_positions() -> Vec<(f32, f32, f32)> {
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
