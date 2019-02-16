// the world is made of multiple chunks.

mod chunk;
use chunk::Chunk;

struct World {
    chunks: Vec<Chunk>
}
