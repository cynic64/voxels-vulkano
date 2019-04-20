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
        World {
            chunks: vec![],       // the chunks will be generated later
            queue
            }
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
            let mut chunk =
                chunk::Chunk::new(self.queue.clone(), coord.clone(), (co_x, co_y, co_z));
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

    pub fn change_coordinate(&mut self, coord: WorldCoordinate, new_state: u8) {
        println!("placing block!");
        // figure out which chunk coord is in by dividing by 32
        let ch_coord = ChunkCoordinate {
            x: (coord.x / 32.0).round() as i32,
            y: (coord.y / 32.0).round() as i32,
            z: (coord.z / 32.0).round() as i32,
        };

        // figure out which idx within the chunk to change by
        // modulo CHUNK_SIZE'ing the xyz of coord and converting
        // that to a linear coordinate
        // we have to add 16 to each coord because chunks are centered on their
        // coordinates
        let subchunk_x = world_coord_to_subchunk_axis(coord.x);
        let subchunk_y = world_coord_to_subchunk_axis(coord.y);
        let subchunk_z = world_coord_to_subchunk_axis(coord.z);
        println!("scx: {}, scy: {}, scz: {}", subchunk_x, subchunk_y, subchunk_z);
        let subchunk_idx = xyz_to_linear(subchunk_x, subchunk_z, subchunk_y);

        // find the idx of the chunk with matching ch_coord
        // and change the correct idx within it
        self.chunks.iter_mut().for_each(|chunk| {
            if chunk.chunk_coord.x == ch_coord.x
                && chunk.chunk_coord.y == ch_coord.y
                && chunk.chunk_coord.z == ch_coord.z
            {
                println!("found a match! placing at {}", subchunk_idx);
                chunk.cells[subchunk_idx] = new_state;
            }
        });
    }

    pub fn update_vbufs(&mut self) {
        // borrow checker still isn't great with closures
        let q = self.queue.clone();

        self.chunks
            .iter_mut()
            .for_each(|chunk| chunk.update_vbuf(q.clone()));
    }

    pub fn get_vbufs(&self) -> Vec<VertexBuffer> {
        self.chunks.iter().map(chunk::Chunk::get_vbuf).collect()
    }

    pub fn generate_nearby_cuboids(&self, camera_pos: Vec3) -> Vec<CuboidOffset> {
        self.chunks
            .iter()
            .map(|chunk| chunk.generate_cuboids_close_to(camera_pos))
            .flatten()
            .collect()
    }
}
