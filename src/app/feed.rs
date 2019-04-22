// Provides a class to make a debug feed that can be drawn on-screen.
// There's the normal debug text that's drawn every frame and has the same
// info each time. However, for events like placing a block, this isn't
// very useful so you can use DebugFeed instead which will make the
// messages scroll into oblivion after a while.

use super::utils::*;

pub struct DebugFeed {
    corner_pos: Point,
    messages: Vec<Message>,
}

struct Message {
    text: String,
}
