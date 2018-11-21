extern crate nalgebra_glm as glm;
extern crate rand;

use self::glm::*;

pub fn get_near_mesh_indices(camera_position: &Vec3, camera_front: &Vec3) -> Vec<usize> {
    use super::SECTOR_SIDE_LEN;
    use super::SIZE;

    let world_size_chunks = SIZE / SECTOR_SIDE_LEN;

    let base_x = (camera_position.x as usize) / SECTOR_SIDE_LEN;
    // z and y are swapped, because for the ca z is up and down and y forward/backward whereas in the camera position it is swapped
    let base_y = (camera_position.z as usize) / SECTOR_SIDE_LEN;
    let base_z = (camera_position.y as usize) / SECTOR_SIDE_LEN;

    // again, the conversion from usize to i32 is kinda shitty :/
    let mut indices = Vec::new();

    let offsets = vec![
        (0, 0, 0),
        (
            round(camera_front.x),
            round(camera_front.z),
            round(camera_front.y),
        ),
    ];

    for offset in offsets.iter() {
        let x = ((base_x as i32) + offset.0) as usize;
        let y = ((base_y as i32) + offset.1) as usize;
        let z = ((base_z as i32) + offset.2) as usize;

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
        -1
    } else if x > 0.5 {
        1
    } else {
        panic!("This shouldn't happen (round)");
    }
}
