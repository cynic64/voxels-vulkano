#[macro_use]
extern crate vulkano;
#[macro_use]
extern crate vulkano_shader_derive;

mod app;
use self::app::App;

fn main() {
    let app = App::new();
    app.run();
}
