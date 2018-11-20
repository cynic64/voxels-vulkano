extern crate nalgebra_glm as glm;
extern crate rand;

use self::glm::*;

#[rustfmt::skip]
const VISIBLE_OFFSETS: [[i32; 3]; 27] = [
    [ 1, -1, -1],
    [ 1,  0, -1],
    [ 1,  1, -1],
    [ 0, -1, -1],
    [ 0,  0, -1],
    [ 0,  1, -1],
    [-1, -1, -1],
    [-1,  0, -1],
    [-1,  1, -1],
    [ 1, -1,  0],
    [ 1,  0,  0],
    [ 1,  1,  0],
    [ 0, -1,  0],
    [ 0,  0,  0],
    [ 0,  1,  0],
    [-1, -1,  0],
    [-1,  0,  0],
    [-1,  1,  0],
    [ 1, -1,  1],
    [ 1,  0,  1],
    [ 1,  1,  1],
    [ 0, -1,  1],
    [ 0,  0,  1],
    [ 0,  1,  1],
    [-1, -1,  1],
    [-1,  0,  1],
    [-1,  1,  1],
];

pub fn get_near_mesh_indices(camera_position: &Vec3) -> Vec<usize> {
    use super::SECTOR_SIDE_LEN;
    use super::SIZE;

    let world_size_chunks = SIZE / SECTOR_SIDE_LEN;

    let base_x = (camera_position.x as usize) / SECTOR_SIDE_LEN;
    let base_y = (camera_position.y as usize) / SECTOR_SIDE_LEN;
    let base_z = (camera_position.z as usize) / SECTOR_SIDE_LEN;

    // again, the conversion from usize to i32 is kinda shitty :/
    let mut indices = Vec::new();

    for offset in VISIBLE_OFFSETS.iter() {
        let x = ((base_x as i32) + offset[0]) as usize;
        let y = ((base_y as i32) + offset[1]) as usize;
        let z = ((base_z as i32) + offset[2]) as usize;

        let linear_idx = z * world_size_chunks * world_size_chunks + y * world_size_chunks + x;
        indices.push(linear_idx);
    }

    indices
}
