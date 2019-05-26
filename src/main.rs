#![feature(test)]
extern crate test;

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

#[cfg(test)]
mod tests {
    use test::Bencher;
    use super::*;

    #[bench]
    fn chunk_generation(b: &mut Bencher) {
        let extensions = vulkano_win::required_extensions();
        let instance = vulkano::instance::Instance::new(None, &extensions, None)
            .expect("failed to create instance");

        let physical = vulkano::instance::PhysicalDevice::from_index(&instance, 0).unwrap();

        let queue_family = physical
            .queue_families()
            .find(|&q| q.supports_graphics())
            .expect("couldn't find a graphical queue family");

        let device_ext = vulkano::device::DeviceExtensions {
            khr_swapchain: true,
            ..vulkano::device::DeviceExtensions::none()
        };

        let (_device, mut queues) = vulkano::device::Device::new(
            physical,
            physical.supported_features(),
            &device_ext,
            [(queue_family, 0.5)].iter().cloned(),
        )
        .expect("failed to create device");
        let queue = queues.next().unwrap();

        b.iter(|| {
            let chunk_coord = utils::ChunkCoordinate {
                x: 0,
                y: 0,
                z: 0,
            };
            let offset = (0.0, 0.0, 0.0);
            let mut chunk = self::app::world::chunk::Chunk::new(queue.clone(), chunk_coord, offset);
            chunk.update_positions();
            chunk.randomize_state();
            chunk.update_vbuf(queue.clone());
        });
    }
}
