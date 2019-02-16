// the world is made of multiple chunks.

pub mod chunk;

use std::sync::Arc;
use super::VertexBuffer;
use nalgebra_glm::Vec3;
use super::RaycastCuboid;

const CHUNK_SIZE: usize = 32;

pub struct World {
    pub chunks: Vec<chunk::Chunk>,
    queue: Arc<vulkano::device::Queue>,
}

impl World {
    pub fn new(queue: Arc<vulkano::device::Queue>) -> Self {
        // create a new chunk
        let mut chunk = chunk::Chunk::new(queue.clone());
        chunk.update_positions();
        chunk.randomize_state();

        World {
            chunks: vec![chunk],
            queue,
        }
    }

    pub fn update_vbufs(&mut self) {
        // borrow checker still isn't great with closures
        let q = self.queue.clone();

        self.chunks.iter_mut().for_each(|x| x.update_vbuf(q.clone()));
    }

    pub fn get_vbuf(&self) -> VertexBuffer {
        // for now, just returns the vbuf of the first chunk.
        self.chunks[0].get_vbuf()
    }

    pub fn generate_nearby_cuboids(&self, camera_pos: Vec3) -> Vec<RaycastCuboid> {
        // again - only returns cuboids from first chunk
        self.chunks[0].generate_cuboids_close_to(camera_pos)
    }
}

pub fn xyz_to_linear(x: usize, y: usize, z: usize) -> usize {
    z * (CHUNK_SIZE * CHUNK_SIZE) + y * CHUNK_SIZE + x
}
