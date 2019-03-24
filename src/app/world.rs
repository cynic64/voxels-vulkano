use super::super::utils::*;

pub mod chunk;

use std::sync::Arc;
use nalgebra_glm::Vec3;

pub struct World {
    pub chunks: Vec<chunk::Chunk>,
    queue: Arc<vulkano::device::Queue>,
}

// chunks will be arranged in Vec<Option<Chunk>>, indexed with [z][y][x] and starting
// at some distance in the negative direction

impl World {
    pub fn new(queue: Arc<vulkano::device::Queue>) -> Self {
        // create a new chunk
        let mut chunk = chunk::Chunk::new(queue.clone(), (0.0, 0.0, 0.0));
        let mut chunk2 = chunk::Chunk::new(queue.clone(), (CHUNK_SIZE as f32, 0.0, 0.0));
        chunk.update_positions();
        chunk.randomize_state();
        chunk2.update_positions();
        chunk2.randomize_state();

        World {
            chunks: vec![chunk, chunk2],
            queue,
        }
    }

    pub fn change_coordinate(&mut self, coord: Coordinate, new_state: u8) {
        // to change a coordinate in the world, we first need to figure
        // out which chunk it's in
        // for now there's only 2 chunks so that's ez
        println!("Bing!");
        let (chunk_idx, subchunk_idx) =
            if coord.x < 32 {
                // it's in the first chunk
                let rel_x = coord.x;
                let rel_y = coord.y;
                let rel_z = coord.z;
                (0, xyz_to_linear(rel_x as usize, rel_z as usize, rel_y as usize))
            } else {
                // it's in the second chunk
                let rel_x = coord.x - 32;
                let rel_y = coord.y;
                let rel_z = coord.z;
                (1, xyz_to_linear(rel_x as usize, rel_z as usize, rel_y as usize))
            };

        self.chunks[chunk_idx].cells[subchunk_idx] = new_state;
    }

    pub fn update_vbufs(&mut self) {
        // borrow checker still isn't great with closures
        let q = self.queue.clone();

        self.chunks.iter_mut().for_each(|ch| ch.update_vbuf(q.clone()));
    }

    pub fn get_vbufs(&self) -> Vec<VertexBuffer> {
        self.chunks.iter().map(|ch| ch.get_vbuf()).collect()
    }

    pub fn generate_nearby_cuboids(&self, camera_pos: Vec3) -> Vec<CuboidOffset> {
        self.chunks.iter().map(|ch| ch.generate_cuboids_close_to(camera_pos)).flatten().collect()
    }
}
