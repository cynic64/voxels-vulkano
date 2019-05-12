use std::sync::Arc;
use vulkano::sync::GpuFuture;

// constants | types
pub const CHUNK_SIZE: usize = 32;
pub type CuboidOffset = na::Isometry3<f32>;

#[derive(Copy, Clone, Debug)]
pub struct Vertex {
    pub position: (f32, f32, f32),
    pub color: (f32, f32, f32, f32),
    pub normal: (f32, f32, f32),
}
impl_vertex!(Vertex, position, color, normal);

pub type VertexBuffer = Arc<vulkano::buffer::immutable::ImmutableBuffer<[Vertex]>>;

// WorldCoordinates are the player's position is 3d space,
// and are the same as camera coordinates
pub struct WorldCoordinate {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

// ChunkCoordinates use the exact same types but instead are used to
// indicate the position of a chunk. You should be able to convert a
// WorldCoordinate to a ChunkCoordinate by dividing by <CHUNK_SIZE>.
#[derive(Debug, Clone)]
pub struct ChunkCoordinate {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

// xy thing for screen coordinates, not vulkan ones.
pub struct ScreenCoordinate {
    pub x: f32,
    pub y: f32,
}

// Struct for sending and recieving information about nearby cuboids.
// One field is a list of isometries for nearby cuboids, and the other
// is a mesh that can be drawn as an overlay for debugging.
pub struct NearbyCuboidsInfo {
    pub overlay_mesh: VertexBuffer,
    pub cuboid_offsets: Vec<CuboidOffset>,
}

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
    y * (CHUNK_SIZE * CHUNK_SIZE) + z * CHUNK_SIZE + x
}

pub fn world_coord_to_subchunk_axis(value: f32) -> usize {
    // first, add 16 because chunks are centered
    let value = value + 16.0;

    if value >= 0.0 {
        // if it's positive, modulo 32 all that's needed
        (value % 32.0) as usize
    } else {
        // otherwise, add 32 until it's positive because
        // rust's modulo is a little weird with negatives
        (value + 32.0 * ((value / -32.0).ceil())) as usize
    }
}
