use super::super::utils::*;

pub mod chunk;

use nalgebra_glm::Vec3;
use std::sync::Arc;

pub struct World {
    pub chunks: Vec<chunk::Chunk>,
    queue: Arc<vulkano::device::Queue>,
}

impl World {
    pub fn new(queue: Arc<vulkano::device::Queue>) -> Self {
        // create a new chunk
        // let mut chunk = chunk::Chunk::new(queue.clone(), (0.0, 0.0, 0.0));
        // let mut chunk2 = chunk::Chunk::new(queue.clone(), (CHUNK_SIZE as f32, 0.0, 0.0));
        // chunk.update_positions();
        // chunk.randomize_state();
        // chunk2.update_positions();
        // chunk2.randomize_state();

        // create an list of Nones as long as TOTAL_CHUNKS
        // the chunks will be generated later
        let chunks = vec![];

        World { chunks, queue }
    }

    pub fn generate_chunk_at(&mut self, coord: ChunkCoordinate) {
        // generates the chunk at coord - if it has not already been generated.

        // check if it already exists
        let mut already_exists = false;
        for chunk in self.chunks.iter() {
            if chunk.chunk_coord.x == coord.x
                && chunk.chunk_coord.y == coord.y
                && chunk.chunk_coord.z == coord.z
            {
                already_exists = true;
            }
        }

        if !already_exists {
            // calculate the chunk's offset from 0 0 0, which will just be
            // the ChunkCoordinate multiplied by CHUNK_SIZE.
            let co_x = (coord.x * 32) as f32;
            let co_y = (coord.y * 32) as f32;
            let co_z = (coord.z * 32) as f32;

            // create the chunk
            let mut chunk = chunk::Chunk::new(self.queue.clone(), coord.clone(), (co_x, co_y, co_z));
            chunk.update_positions();
            chunk.randomize_state();

            // add it to our chunk list
            self.chunks.push(chunk);

            println!(
                "Adding chunk at chunk coordinates {:?}. Total # of chunks: {}",
                coord,
                self.chunks.len()
            );
        }
    }

    pub fn change_coordinate(&mut self, _coord: WorldCoordinate, _new_state: u8) {
        // this doesn't work for now, ok
        println!("Change_coordinate unimplemented.");
    }

    pub fn update_vbufs(&mut self) {
        // borrow checker still isn't great with closures
        let q = self.queue.clone();

        self.chunks
            .iter_mut()
            .for_each(|chunk| chunk.update_vbuf(q.clone()));
    }

    pub fn get_vbufs(&self) -> Vec<VertexBuffer> {
        self.chunks.iter().map(|chunk| chunk.get_vbuf()).collect()
    }

    pub fn generate_nearby_cuboids(&self, camera_pos: Vec3) -> Vec<CuboidOffset> {
        self.chunks
            .iter()
            .map(|chunk| chunk.generate_cuboids_close_to(camera_pos))
            .flatten()
            .collect()
    }
}
