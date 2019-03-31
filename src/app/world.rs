use super::super::utils::*;

pub mod chunk;

use nalgebra_glm::Vec3;
use std::sync::Arc;

pub struct World {
    pub chunks: Vec<Option<chunk::Chunk>>,
    queue: Arc<vulkano::device::Queue>,
}

// how far away from the player chunks will still be rendered (in chunks)
const VIEW_DIST: u32 = 8;
// the side length of the cube of visible chunks
const VIEW_LEN: u32 = VIEW_DIST * 2 + 1;
// total number of chunks that the chunk array will have space for
const TOTAL_CHUNKS: u32 = VIEW_LEN * VIEW_LEN * VIEW_LEN;

// chunks will be arranged in Vec<Option<Chunk>>, indexed with [z][y][x] and starting
// VIEW_DIST in the negative direction.

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
        let chunks = (0..TOTAL_CHUNKS).map(|_| None).collect();

        World { chunks, queue }
    }

    pub fn generate_chunk_at(&mut self, coord: ChunkCoordinate) {
        // generates the chunk at coord - if it has not already been generated.

        // calculate the chunk's index.
        // I really need to make a generic function for this kind of stuff
        // need better variable names! :(
        // and maybe check for overflow
        let o_x = (coord.x + (VIEW_DIST as i32)) as u32;
        let o_y = (coord.y + (VIEW_DIST as i32)) as u32;
        let o_z = (coord.z + (VIEW_DIST as i32)) as u32;
        let chunk_idx = (o_z * VIEW_DIST * VIEW_DIST + o_y * VIEW_DIST + o_x) as usize;
        if self.chunks[chunk_idx].is_none() {
            // calculate the chunk's offset from 0 0 0, which will just be
            // the ChunkCoordinate multiplied by CHUNK_SIZE.
            let co_x = (coord.x * 32) as f32;
            let co_y = (coord.y * 32) as f32;
            let co_z = (coord.z * 32) as f32;

            // create the chunk
            let mut chunk = chunk::Chunk::new(self.queue.clone(), (co_x, co_y, co_z));
            chunk.update_positions();
            chunk.randomize_state();

            // add it to our chunk list
            self.chunks[chunk_idx] = Some(chunk);
        }
    }

    pub fn change_coordinate(&mut self, coord: WorldCoordinate, new_state: u8) {
        // to change a coordinate in the world, we first need to figure
        // out which chunk it's in
        // for now there's only 2 chunks so that's ez
        let (chunk_idx, subchunk_idx) = if coord.x < 32 {
            // it's in the first chunk
            let rel_x = coord.x;
            let rel_y = coord.y;
            let rel_z = coord.z;
            (
                0,
                xyz_to_linear(rel_x as usize, rel_z as usize, rel_y as usize),
            )
        } else {
            // it's in the second chunk
            let rel_x = coord.x - 32;
            let rel_y = coord.y;
            let rel_z = coord.z;
            (
                1,
                xyz_to_linear(rel_x as usize, rel_z as usize, rel_y as usize),
            )
        };

        // make sure we're not trying to place a block in a None chunk
        let chunk = &mut self.chunks[chunk_idx];
        match chunk {
            Some(ch) => ch.cells[subchunk_idx] = new_state,
            None => println!("You just tried to place a block in a chunk that doesn't exist. This bit needs to be implemented, cause in that case the chunk should be generated so a block can be placed.")
        };
    }

    pub fn update_vbufs(&mut self) {
        // borrow checker still isn't great with closures
        let q = self.queue.clone();

        self.chunks.iter_mut().for_each(|chunk| {
            // only update vbufs of chunks that aren't None
            match chunk {
                Some(ch) => ch.update_vbuf(q.clone()),
                None => {}
            }
        });
    }

    pub fn get_vbufs(&self) -> Vec<VertexBuffer> {
        self.chunks
            .iter()
            .filter_map(|chunk| match chunk {
                Some(ch) => Some(ch.get_vbuf()),
                None => None,
            })
            .collect()
    }

    pub fn generate_nearby_cuboids(&self, camera_pos: Vec3) -> Vec<CuboidOffset> {
        self.chunks
            .iter()
            .filter_map(|chunk| {
                match chunk {
                    Some(ch) => Some(ch.generate_cuboids_close_to(camera_pos)),
                    None => None
                }
            })
            .flatten()
            .collect()
    }
}
