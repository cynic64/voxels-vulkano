extern crate nalgebra_glm as glm;
extern crate rand;

use self::glm::*;

const VISIBLE_OFFSETS: [(i32, i32, i32); 125] = [
    (-2, -2, -2),
    (-1, -2, -2),
    (0, -2, -2),
    (1, -2, -2),
    (2, -2, -2),
    (-2, -1, -2),
    (-1, -1, -2),
    (0, -1, -2),
    (1, -1, -2),
    (2, -1, -2),
    (-2, 0, -2),
    (-1, 0, -2),
    (0, 0, -2),
    (1, 0, -2),
    (2, 0, -2),
    (-2, 1, -2),
    (-1, 1, -2),
    (0, 1, -2),
    (1, 1, -2),
    (2, 1, -2),
    (-2, 2, -2),
    (-1, 2, -2),
    (0, 2, -2),
    (1, 2, -2),
    (2, 2, -2),
    (-2, -2, -1),
    (-1, -2, -1),
    (0, -2, -1),
    (1, -2, -1),
    (2, -2, -1),
    (-2, -1, -1),
    (-1, -1, -1),
    (0, -1, -1),
    (1, -1, -1),
    (2, -1, -1),
    (-2, 0, -1),
    (-1, 0, -1),
    (0, 0, -1),
    (1, 0, -1),
    (2, 0, -1),
    (-2, 1, -1),
    (-1, 1, -1),
    (0, 1, -1),
    (1, 1, -1),
    (2, 1, -1),
    (-2, 2, -1),
    (-1, 2, -1),
    (0, 2, -1),
    (1, 2, -1),
    (2, 2, -1),
    (-2, -2, 0),
    (-1, -2, 0),
    (0, -2, 0),
    (1, -2, 0),
    (2, -2, 0),
    (-2, -1, 0),
    (-1, -1, 0),
    (0, -1, 0),
    (1, -1, 0),
    (2, -1, 0),
    (-2, 0, 0),
    (-1, 0, 0),
    (0, 0, 0),
    (1, 0, 0),
    (2, 0, 0),
    (-2, 1, 0),
    (-1, 1, 0),
    (0, 1, 0),
    (1, 1, 0),
    (2, 1, 0),
    (-2, 2, 0),
    (-1, 2, 0),
    (0, 2, 0),
    (1, 2, 0),
    (2, 2, 0),
    (-2, -2, 1),
    (-1, -2, 1),
    (0, -2, 1),
    (1, -2, 1),
    (2, -2, 1),
    (-2, -1, 1),
    (-1, -1, 1),
    (0, -1, 1),
    (1, -1, 1),
    (2, -1, 1),
    (-2, 0, 1),
    (-1, 0, 1),
    (0, 0, 1),
    (1, 0, 1),
    (2, 0, 1),
    (-2, 1, 1),
    (-1, 1, 1),
    (0, 1, 1),
    (1, 1, 1),
    (2, 1, 1),
    (-2, 2, 1),
    (-1, 2, 1),
    (0, 2, 1),
    (1, 2, 1),
    (2, 2, 1),
    (-2, -2, 2),
    (-1, -2, 2),
    (0, -2, 2),
    (1, -2, 2),
    (2, -2, 2),
    (-2, -1, 2),
    (-1, -1, 2),
    (0, -1, 2),
    (1, -1, 2),
    (2, -1, 2),
    (-2, 0, 2),
    (-1, 0, 2),
    (0, 0, 2),
    (1, 0, 2),
    (2, 0, 2),
    (-2, 1, 2),
    (-1, 1, 2),
    (0, 1, 2),
    (1, 1, 2),
    (2, 1, 2),
    (-2, 2, 2),
    (-1, 2, 2),
    (0, 2, 2),
    (1, 2, 2),
    (2, 2, 2),
];

pub fn get_near_mesh_indices(camera_position: &Vec3, camera_front: &Vec3) -> Vec<usize> {
    use super::SECTOR_SIDE_LEN;
    use super::SIZE;

    let world_size_chunks = SIZE / SECTOR_SIDE_LEN;

    let mut base_x = ((camera_position.x as usize) / SECTOR_SIDE_LEN) as i32;
    let mut base_y = ((camera_position.z as usize) / SECTOR_SIDE_LEN) as i32;
    let mut base_z = ((camera_position.y as usize) / SECTOR_SIDE_LEN) as i32;

    base_x += round(camera_front.x);
    base_y += round(camera_front.z);
    base_z += round(camera_front.y);

    // again, the conversion from usize to i32 is kinda shitty :/
    let mut indices = Vec::new();

    for &offset in VISIBLE_OFFSETS.iter() {
        let x = (base_x + offset.0) as usize;
        let y = (base_y + offset.1) as usize;
        let z = (base_z + offset.2) as usize;

        let linear_idx = z * world_size_chunks * world_size_chunks + y * world_size_chunks + x;
        indices.push(linear_idx);
    }

    indices
}

fn round(x: f32) -> i32 {
    // idk if there is a std function for this
    if x >= -0.5 && x <= 0.5 {
        0
    } else if x < -0.5 {
        -2
    } else if x > 0.5 {
        2
    } else {
        panic!("This shouldn't happen (round)");
    }
}
