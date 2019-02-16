use std::sync::Arc;
use vulkano::sync::GpuFuture;
use na::Isometry3;

// constants | types
pub const CHUNK_SIZE: usize = 32;

#[derive(Copy, Clone, Debug)]
pub struct Vertex {
    pub position: (f32, f32, f32),
    pub color: (f32, f32, f32, f32),
    pub normal: (f32, f32, f32),
}
impl_vertex!(Vertex, position, color, normal);

pub type VertexBuffer = Arc<vulkano::buffer::immutable::ImmutableBuffer<[Vertex]>>;

pub type RaycastCuboid = (Isometry3<f32>, usize);


// functions
pub fn vbuf_from_verts(queue: Arc<vulkano::device::Queue>, vertices: Vec<Vertex>) -> VertexBuffer {
    let (buffer, future) = vulkano::buffer::immutable::ImmutableBuffer::from_iter(
        vertices.iter().cloned(),
        vulkano::buffer::BufferUsage::vertex_buffer(),
        queue.clone(),
    )
    .unwrap();
    future.flush().unwrap();

    buffer
}

pub fn get_elapsed(start: std::time::Instant) -> f32 {
    start.elapsed().as_secs() as f32 + start.elapsed().subsec_nanos() as f32 / 1_000_000_000.0
}

pub fn make_empty_vbuf(queue: Arc<vulkano::device::Queue>) -> VertexBuffer {
    vbuf_from_verts(queue, vec![])
}

pub fn xyz_to_linear(x: usize, y: usize, z: usize) -> usize {
    z * (CHUNK_SIZE * CHUNK_SIZE) + y * CHUNK_SIZE + x
}
