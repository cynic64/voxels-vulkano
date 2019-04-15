#[macro_use]
extern crate vulkano;
#[macro_use]
extern crate vulkano_shader_derive;

extern crate nalgebra as na;
extern crate ncollide3d;

mod app;
use self::app::App;

pub mod utils;

fn main() {
    let mut app = App::new();
    app.run();
}
