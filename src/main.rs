// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

extern crate nalgebra_glm as glm;
extern crate rand;
extern crate time;
extern crate winit;

#[macro_use]
extern crate vulkano;
#[macro_use]
extern crate vulkano_shader_derive;
extern crate vulkano_win;
use winit::{Event, EventsLoop, KeyboardInput, VirtualKeyCode, WindowBuilder, WindowEvent, ElementState};

// my stuff
mod ca;
mod camera;

use vulkano::sync::GpuFuture;
use vulkano_win::VkSurfaceBuild;

use std::sync::Arc;

const SIZE: usize = 64;

#[derive(Copy, Clone, Debug)]
struct Vertex {
    position: (f32, f32, f32),
    color: (f32, f32, f32, f32),
}

impl_vertex!(Vertex, position, color);

#[rustfmt::skip]
const CUBE_VERTICES: [Vertex; 8] = [
    Vertex { position: (-0.5, -0.5, -0.5), color: (1.0, 1.0, 1.0, 1.0) },
    Vertex { position: ( 0.5, -0.5, -0.5), color: (0.0, 1.0, 1.0, 1.0) },
    Vertex { position: ( 0.5,  0.5, -0.5), color: (1.0, 0.0, 1.0, 1.0) },
    Vertex { position: (-0.5,  0.5, -0.5), color: (1.0, 1.0, 0.0, 1.0) },
    Vertex { position: (-0.5, -0.5,  0.5), color: (0.0, 0.0, 1.0, 1.0) },
    Vertex { position: ( 0.5, -0.5,  0.5), color: (1.0, 0.0, 0.0, 1.0) },
    Vertex { position: ( 0.5,  0.5,  0.5), color: (0.0, 1.0, 0.0, 1.0) },
    Vertex { position: (-0.5,  0.5,  0.5), color: (0.0, 0.0, 0.0, 1.0) }
];

const CUBE_INDICES: [u32; 36] = [
    0, 1, 3, 3, 1, 2, 1, 5, 2, 2, 5, 6, 5, 4, 6, 6, 4, 7, 4, 0, 7, 7, 0, 3, 3, 2, 7, 7, 2, 6, 4, 5,
    0, 0, 5, 1,
];

fn main() {
    let positions = generate_positions();
    let mut ca = setup_ca();
    let mut cam = camera::Camera::default();

    //-------------------------------------------------------------------------------------//
    let extensions = vulkano_win::required_extensions();
    let instance = vulkano::instance::Instance::new(None, &extensions, None)
        .expect("failed to create instance");

    let physical = vulkano::instance::PhysicalDevice::enumerate(&instance)
        .next()
        .expect("no device available");
    println!(
        "Using device: {} (type: {:?})",
        physical.name(),
        physical.ty()
    );

    let mut events_loop = winit::EventsLoop::new();
    let surface = WindowBuilder::new()
        .build_vk_surface(&events_loop, instance.clone())
        .unwrap();

    let window = surface.window();
    window.hide_cursor(true);

    let mut dimensions;

    let queue_family = physical
        .queue_families()
        .find(|&q| q.supports_graphics() && surface.is_supported(q).unwrap_or(false))
        .expect("couldn't find a graphical queue family");

    let device_ext = vulkano::device::DeviceExtensions {
        khr_swapchain: true,
        ..vulkano::device::DeviceExtensions::none()
    };

    let (device, mut queues) = vulkano::device::Device::new(
        physical,
        physical.supported_features(),
        &device_ext,
        [(queue_family, 0.5)].iter().cloned(),
    )
    .expect("failed to create device");
    let queue = queues.next().unwrap();

    let (mut swapchain, mut images) = {
        let caps = surface
            .capabilities(physical)
            .expect("failed to get surface capabilities");

        dimensions = caps.current_extent.unwrap_or([1024, 768]);

        let usage = caps.supported_usage_flags;
        let format = caps.supported_formats[0].0;
        let alpha = caps.supported_composite_alpha.iter().next().unwrap();

        vulkano::swapchain::Swapchain::new(
            device.clone(),
            surface.clone(),
            caps.min_image_count,
            format,
            dimensions,
            1,
            usage,
            &queue,
            vulkano::swapchain::SurfaceTransform::Identity,
            alpha,
            vulkano::swapchain::PresentMode::Fifo,
            true,
            None,
        )
        .expect("failed to create swapchain")
    };

    let mut depth_buffer = vulkano::image::attachment::AttachmentImage::transient(
        device.clone(),
        dimensions,
        vulkano::format::D16Unorm,
    )
    .unwrap();

    // mvp
    let model = glm::scale(
        &glm::Mat4::identity(),
        &glm::vec3(1.0, 1.0, 1.0),
        );
    let mut view = glm::look_at(
        &glm::vec3(1., 0., 1.),
        &glm::vec3(0., 0., 0.),
        &glm::vec3(0., 1., 0.)
        );
    let projection = glm::perspective(
        // fov
        1.5,
        // aspect ratio
        16. / 9.,
        // near
        0.0001,
        // far
        100_000.
        );

    let (mut vertex_buffer, indices) = update_vbuf(&ca.cells, &positions, device.clone());
    let index_buffer = vulkano::buffer::cpu_access::CpuAccessibleBuffer::from_iter(
        device.clone(),
        vulkano::buffer::BufferUsage::all(),
        indices.iter().cloned(),
    )
    .expect("failed to create buffer");

    let uniform_buffer = vulkano::buffer::cpu_pool::CpuBufferPool::<vs::ty::Data>::new(
        device.clone(),
        vulkano::buffer::BufferUsage::all(),
    );

    let vs = vs::Shader::load(device.clone()).expect("failed to create shader module");
    let fs = fs::Shader::load(device.clone()).expect("failed to create shader module");

    let renderpass = Arc::new(
        single_pass_renderpass!(device.clone(),
            attachments: {
                color: {
                    load: Clear,
                    store: Store,
                    format: swapchain.format(),
                    samples: 1,
                },
                depth: {
                    load: Clear,
                    store: DontCare,
                    format: vulkano::format::Format::D16Unorm,
                    samples: 1,
                }
            },
            pass: {
                color: [color],
                depth_stencil: {depth}
            }
        )
        .unwrap(),
    );

    let pipeline = Arc::new(
        vulkano::pipeline::GraphicsPipeline::start()
            .vertex_input_single_buffer()
            .vertex_shader(vs.main_entry_point(), ())
            .triangle_list()
            .viewports_dynamic_scissors_irrelevant(1)
            .fragment_shader(fs.main_entry_point(), ())
            .depth_stencil_simple_depth()
            .render_pass(vulkano::framebuffer::Subpass::from(renderpass.clone(), 0).unwrap())
            .build(device.clone())
            .unwrap(),
    );
    let mut framebuffers: Option<Vec<Arc<vulkano::framebuffer::Framebuffer<_, _>>>> = None;

    let mut recreate_swapchain = false;

    let mut previous_frame = Box::new(vulkano::sync::now(device.clone())) as Box<GpuFuture>;
    let rotation_start = std::time::Instant::now();

    let mut dynamic_state = vulkano::command_buffer::DynamicState {
        line_width: None,
        viewports: Some(vec![vulkano::pipeline::viewport::Viewport {
            origin: [0.0, 0.0],
            dimensions: [dimensions[0] as f32, dimensions[1] as f32],
            depth_range: 0.0..1.0,
        }]),
        scissors: None,
    };

    // mainloop
    let mut frame_count = 0;
    struct KeysPressed {
        w: bool, a: bool, s: bool, d: bool
    }
    let mut keys_pressed = KeysPressed { w: false, a: false, s: false, d: false };
    let mut last_frame = std::time::Instant::now();

    loop {
        let delta = get_elapsed(last_frame);
        last_frame = std::time::Instant::now();

        frame_count += 1;
        previous_frame.cleanup_finished();

        if recreate_swapchain {
            dimensions = surface
                .capabilities(physical)
                .expect("failed to get surface capabilities")
                .current_extent
                .unwrap_or([1024, 768]);

            let (new_swapchain, new_images) = match swapchain.recreate_with_dimension(dimensions) {
                Ok(r) => r,
                Err(vulkano::swapchain::SwapchainCreationError::UnsupportedDimensions) => {
                    continue;
                }
                Err(err) => panic!("{:?}", err),
            };

            swapchain = new_swapchain;
            images = new_images;

            depth_buffer = vulkano::image::attachment::AttachmentImage::transient(
                device.clone(),
                dimensions,
                vulkano::format::D16Unorm,
            )
            .unwrap();

            framebuffers = None;

            // todo: fix aspect ratio on resize
            dynamic_state.viewports = Some(vec![vulkano::pipeline::viewport::Viewport {
                origin: [0.0, 0.0],
                dimensions: [dimensions[0] as f32, dimensions[1] as f32],
                depth_range: 0.0..1.0,
            }]);

            recreate_swapchain = false;
        }

        if framebuffers.is_none() {
            framebuffers = Some(
                images
                    .iter()
                    .map(|image| {
                        Arc::new(
                            vulkano::framebuffer::Framebuffer::start(renderpass.clone())
                                .add(image.clone())
                                .unwrap()
                                .add(depth_buffer.clone())
                                .unwrap()
                                .build()
                                .unwrap(),
                        )
                    })
                    .collect::<Vec<_>>(),
            );
        }

        // camera
        if keys_pressed.w { cam.move_forward(delta); }
        if keys_pressed.s { cam.move_backward(delta); }
        if keys_pressed.a { cam.move_left(delta); }
        if keys_pressed.d { cam.move_right(delta); }

        view = cam.get_view_matrix().into();

        let uniform_buffer_subbuffer = {
            let uniform_data = vs::ty::Data {
                world: model.into(),
                view: view.into(),
                proj: projection.into(),
            };

            uniform_buffer.next(uniform_data).unwrap()
        };

        let set = Arc::new(
            vulkano::descriptor::descriptor_set::PersistentDescriptorSet::start(
                pipeline.clone(),
                0,
            )
            .add_buffer(uniform_buffer_subbuffer)
            .unwrap()
            .build()
            .unwrap(),
        );

        let (image_num, acquire_future) =
            match vulkano::swapchain::acquire_next_image(swapchain.clone(), None) {
                Ok(r) => r,
                Err(vulkano::swapchain::AcquireError::OutOfDate) => {
                    recreate_swapchain = true;
                    continue;
                }
                Err(err) => panic!("{:?}", err),
            };

        let command_buffer =
            vulkano::command_buffer::AutoCommandBufferBuilder::primary_one_time_submit(
                device.clone(),
                queue.family(),
            )
            .unwrap()
            .begin_render_pass(
                framebuffers.as_ref().unwrap()[image_num].clone(),
                false,
                vec![[0.2, 0.2, 0.2, 1.0].into(), 1f32.into()],
            )
            .unwrap()
            .draw_indexed(
                pipeline.clone(),
                &dynamic_state,
                vertex_buffer.clone(),
                index_buffer.clone(),
                set.clone(),
                (),
            )
            .unwrap()
            .end_render_pass()
            .unwrap()
            .build()
            .unwrap();

        let future = previous_frame
            .join(acquire_future)
            .then_execute(queue.clone(), command_buffer)
            .unwrap()
            .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
            .then_signal_fence_and_flush();

        match future {
            Ok(future) => {
                previous_frame = Box::new(future) as Box<_>;
            }
            Err(vulkano::sync::FlushError::OutOfDate) => {
                recreate_swapchain = true;
                previous_frame = Box::new(vulkano::sync::now(device.clone())) as Box<_>;
            }
            Err(e) => {
                println!("{:?}", e);
                previous_frame = Box::new(vulkano::sync::now(device.clone())) as Box<_>;
            }
        }

        let mut done = false;
        events_loop.poll_events(|event| {
            if let winit::Event::WindowEvent { event, .. } = event {
                match event {
                    WindowEvent::CloseRequested => done = true,
                    WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                virtual_keycode: Some(VirtualKeyCode::Escape),
                                ..
                            },
                        ..
                    } => done = true,
                    WindowEvent::KeyboardInput { input: KeyboardInput { virtual_keycode: Some(VirtualKeyCode::N), state: ElementState::Pressed, .. }, .. } => {
                        ca.next_gen();
                        vertex_buffer = update_vbuf(&ca.cells, &positions, device.clone()).0;
                    },
                    WindowEvent::KeyboardInput { input: KeyboardInput { virtual_keycode: Some(VirtualKeyCode::W), state: ElementState::Pressed, .. }, .. } => { keys_pressed.w = true; },
                    WindowEvent::KeyboardInput { input: KeyboardInput { virtual_keycode: Some(VirtualKeyCode::A), state: ElementState::Pressed, .. }, .. } => { keys_pressed.a = true; },
                    WindowEvent::KeyboardInput { input: KeyboardInput { virtual_keycode: Some(VirtualKeyCode::S), state: ElementState::Pressed, .. }, .. } => { keys_pressed.s = true; },
                    WindowEvent::KeyboardInput { input: KeyboardInput { virtual_keycode: Some(VirtualKeyCode::D), state: ElementState::Pressed, .. }, .. } => { keys_pressed.d = true; },
                    WindowEvent::KeyboardInput { input: KeyboardInput { virtual_keycode: Some(VirtualKeyCode::W), state: ElementState::Released,.. }, .. } => { keys_pressed.w =false; },
                    WindowEvent::KeyboardInput { input: KeyboardInput { virtual_keycode: Some(VirtualKeyCode::A), state: ElementState::Released,.. }, .. } => { keys_pressed.a =false; },
                    WindowEvent::KeyboardInput { input: KeyboardInput { virtual_keycode: Some(VirtualKeyCode::S), state: ElementState::Released,.. }, .. } => { keys_pressed.s =false; },
                    WindowEvent::KeyboardInput { input: KeyboardInput { virtual_keycode: Some(VirtualKeyCode::D), state: ElementState::Released,.. }, .. } => { keys_pressed.d =false; },

                    WindowEvent::CursorMoved { position: p, .. } => {
                        let (x_diff, y_diff) = (p.x - (dimensions[0] as f64 / 2.0), (p.y - dimensions[1] as f64 / 2.0));
                        cam.mouse_move(x_diff as f32, y_diff as f32);
                        window.set_cursor_position(winit::dpi::LogicalPosition { x: dimensions[0] as f64 / 2.0, y: dimensions[1] as f64 / 2.0 })
                            .expect("Couldn't re-set cursor position!");
                    },
                    _ => {}
                }
            }
        });

        if done {
            break;
        }
    }

    println!("Frames: {}", frame_count);
}

fn generate_vertices(cells: &[u8], positions: &[(f32, f32, f32)]) -> Vec<Vertex> {
    positions
        .iter()
        .enumerate()
        .map(|(idx, &offset)| {
            let mut color;
            if (idx > SIZE * SIZE + SIZE)
                && (idx
                    < (SIZE * SIZE * SIZE)
                        - (SIZE * SIZE)
                        - SIZE
                        - 1)
            {
                let cur_state = cells[idx];
                let neighbors = [
                    cells[idx + (SIZE * SIZE) + SIZE + 1],
                    cells[idx + (SIZE * SIZE) + SIZE],
                    cells[idx + (SIZE * SIZE) + SIZE - 1],
                    cells[idx + (SIZE * SIZE) + 1],
                    cells[idx + (SIZE * SIZE)],
                    cells[idx + (SIZE * SIZE) - 1],
                    cells[idx + (SIZE * SIZE) - SIZE + 1],
                    cells[idx + (SIZE * SIZE) - SIZE],
                    cells[idx + (SIZE * SIZE) - SIZE - 1],
                    cells[idx + SIZE + 1],
                    cells[idx + SIZE],
                    cells[idx + SIZE - 1],
                    cells[idx + 1],
                    cells[idx - 1],
                    cells[idx - SIZE + 1],
                    cells[idx - SIZE],
                    cells[idx - SIZE - 1],
                    cells[idx - (SIZE * SIZE) + SIZE + 1],
                    cells[idx - (SIZE * SIZE) + SIZE],
                    cells[idx - (SIZE * SIZE) + SIZE - 1],
                    cells[idx - (SIZE * SIZE) + 1],
                    cells[idx - (SIZE * SIZE)],
                    cells[idx - (SIZE * SIZE) - 1],
                    cells[idx - (SIZE * SIZE) - SIZE + 1],
                    cells[idx - (SIZE * SIZE) - SIZE],
                    cells[idx - (SIZE * SIZE) - SIZE - 1],
                ];

                let count: u8 = neighbors.iter().sum();
                let value = 1.0 - ((count as f32) / 26.0);
                color = (value, value, value, 1.0);
            } else {
                color = (1.0, 0.0, 0.0, 1.0);
            }

            CUBE_VERTICES.iter().map(move |v| {
                let pos = v.position;
                Vertex {
                    position: (pos.0 + offset.0, pos.1 + offset.1, pos.2 + offset.2),
                    color,
                }
            })
        })
        .flatten()
        .collect()
}

fn generate_indices(cells: &[u8]) -> Vec<u32> {
    cells
        .iter()
        .enumerate()
        .filter_map(|(idx, &state)| {
            if state > 0 {
                // make sure this cell isn't totally obscured
                let neighbors: u8 = [
                    cells[idx + (SIZE * SIZE) + SIZE + 1],
                    cells[idx + (SIZE * SIZE) + SIZE],
                    cells[idx + (SIZE * SIZE) + SIZE - 1],
                    cells[idx + (SIZE * SIZE) + 1],
                    cells[idx + (SIZE * SIZE)],
                    cells[idx + (SIZE * SIZE) - 1],
                    cells[idx + (SIZE * SIZE) - SIZE + 1],
                    cells[idx + (SIZE * SIZE) - SIZE],
                    cells[idx + (SIZE * SIZE) - SIZE - 1],
                    cells[idx + SIZE + 1],
                    cells[idx + SIZE],
                    cells[idx + SIZE - 1],
                    cells[idx + 1],
                    cells[idx - 1],
                    cells[idx - SIZE + 1],
                    cells[idx - SIZE],
                    cells[idx - SIZE - 1],
                    cells[idx - (SIZE * SIZE) + SIZE + 1],
                    cells[idx - (SIZE * SIZE) + SIZE],
                    cells[idx - (SIZE * SIZE) + SIZE - 1],
                    cells[idx - (SIZE * SIZE) + 1],
                    cells[idx - (SIZE * SIZE)],
                    cells[idx - (SIZE * SIZE) - 1],
                    cells[idx - (SIZE * SIZE) - SIZE + 1],
                    cells[idx - (SIZE * SIZE) - SIZE],
                    cells[idx - (SIZE * SIZE) - SIZE - 1],
                ]
                .iter()
                .sum();

                if neighbors < 26 {
                    let start_idx = idx * CUBE_VERTICES.len();
                    Some(
                        CUBE_INDICES
                            .iter()
                            .map(|c_idx| (start_idx as u32) + c_idx)
                            .collect::<Vec<_>>(),
                    )
                } else {
                    None
                }
            } else {
                None
            }
        })
        .flatten()
        .collect()
}

fn generate_positions() -> Vec<(f32, f32, f32)> {
    (0..SIZE)
        .map(|y| {
            (0..SIZE)
                .map(|z| {
                    (0..SIZE)
                        .map(|x| (x as f32, y as f32, z as f32))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        })
        .flatten()
        .flatten()
        .collect()
}

fn setup_ca() -> ca::CellA {
    let mut ca = ca::CellA::new(SIZE, SIZE, SIZE, 13, 26, 14, 26);
    ca.randomize();
    for _ in 0..20 {
        ca.next_gen()
    }

    ca
}

fn update_vbuf(
    cells: &[u8],
    positions: &[(f32, f32, f32)],
    device: std::sync::Arc<vulkano::device::Device>,
) -> (
    std::sync::Arc<vulkano::buffer::cpu_access::CpuAccessibleBuffer<[Vertex]>>,
    Vec<u32>,
) {
    let vertices = generate_vertices(cells, positions);
    let vertex_buffer = vulkano::buffer::cpu_access::CpuAccessibleBuffer::from_iter(
        device.clone(),
        vulkano::buffer::BufferUsage::vertex_buffer(),
        vertices.iter().cloned(),
    )
    .expect("failed to create buffer");

    let indices = generate_indices(cells);

    (vertex_buffer, indices)
}

pub fn get_elapsed ( start: std::time::Instant ) -> f32 {
    start.elapsed().as_secs() as f32 + start.elapsed().subsec_millis() as f32 / 1000.0
}

mod vs {
    #[derive(VulkanoShader)]
    #[ty = "vertex"]
    #[src = "
#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec4 color;

layout(location = 0) out vec4 v_color;

layout(set = 0, binding = 0) uniform Data {
    mat4 world;
    mat4 view;
    mat4 proj;
} uniforms;

void main() {
    v_color = color;
    mat4 worldview = uniforms.view * uniforms.world;
    gl_Position = uniforms.proj * worldview * vec4(position, 1.0);
}
"]
    #[allow(dead_code)]
    struct Dummy;
}

mod fs {
    #[derive(VulkanoShader)]
    #[ty = "fragment"]
    #[src = "
#version 450

layout(location = 0) in vec4 v_color;
layout(location = 0) out vec4 f_color;

void main() {
    f_color = v_color;
}
"]
    #[allow(dead_code)]
    struct Dummy;
}
