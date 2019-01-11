fn main() {
    let app = App::new();
    app.say_hello();
}

struct App {}

impl App {
    fn new() -> App {
        App {}
    }

    fn say_hello(&self) {
        println!("Hello!");
    }
}
