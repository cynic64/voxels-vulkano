// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

extern crate cgmath;
extern crate rand;
extern crate time;
extern crate winit;

#[macro_use]
extern crate vulkano;
#[macro_use]
extern crate vulkano_shader_derive;
extern crate vulkano_win;

// my stuff
mod ca;

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
const CUBE_VERTICES: [Vertex; 36] = [
    Vertex { position: (-0.5, -0.5, -0.5), color: (1.0, 0.0, 0.0, 1.0), },
    Vertex { position: ( 0.5, -0.5, -0.5), color: (1.0, 0.0, 0.0, 1.0), },
    Vertex { position: ( 0.5,  0.5, -0.5), color: (1.0, 0.0, 0.0, 1.0), },
    Vertex { position: ( 0.5,  0.5, -0.5), color: (1.0, 0.0, 0.0, 1.0), },
    Vertex { position: (-0.5,  0.5, -0.5), color: (1.0, 0.0, 0.0, 1.0), },
    Vertex { position: (-0.5, -0.5, -0.5), color: (1.0, 0.0, 0.0, 1.0), },
    Vertex { position: (-0.5, -0.5,  0.5), color: (0.0, 1.0, 0.0, 1.0), },
    Vertex { position: ( 0.5,  0.5,  0.5), color: (0.0, 1.0, 0.0, 1.0), },
    Vertex { position: ( 0.5, -0.5,  0.5), color: (0.0, 1.0, 0.0, 1.0), },
    Vertex { position: ( 0.5,  0.5,  0.5), color: (0.0, 1.0, 0.0, 1.0), },
    Vertex { position: (-0.5, -0.5,  0.5), color: (0.0, 1.0, 0.0, 1.0), },
    Vertex { position: (-0.5,  0.5,  0.5), color: (0.0, 1.0, 0.0, 1.0), },
    Vertex { position: (-0.5,  0.5,  0.5), color: (0.0, 0.0, 1.0, 1.0), },
    Vertex { position: (-0.5, -0.5, -0.5), color: (0.0, 0.0, 1.0, 1.0), },
    Vertex { position: (-0.5,  0.5, -0.5), color: (0.0, 0.0, 1.0, 1.0), },
    Vertex { position: (-0.5, -0.5, -0.5), color: (0.0, 0.0, 1.0, 1.0), },
    Vertex { position: (-0.5,  0.5,  0.5), color: (0.0, 0.0, 1.0, 1.0), },
    Vertex { position: (-0.5, -0.5,  0.5), color: (0.0, 0.0, 1.0, 1.0), },
    Vertex { position: ( 0.5,  0.5,  0.5), color: (1.0, 1.0, 0.0, 1.0), },
    Vertex { position: ( 0.5,  0.5, -0.5), color: (1.0, 1.0, 0.0, 1.0), },
    Vertex { position: ( 0.5, -0.5, -0.5), color: (1.0, 1.0, 0.0, 1.0), },
    Vertex { position: ( 0.5, -0.5, -0.5), color: (1.0, 1.0, 0.0, 1.0), },
    Vertex { position: ( 0.5, -0.5,  0.5), color: (1.0, 1.0, 0.0, 1.0), },
    Vertex { position: ( 0.5,  0.5,  0.5), color: (1.0, 1.0, 0.0, 1.0), },
    Vertex { position: (-0.5, -0.5, -0.5), color: (1.0, 0.0, 1.0, 1.0), },
    Vertex { position: ( 0.5, -0.5,  0.5), color: (1.0, 0.0, 1.0, 1.0), },
    Vertex { position: ( 0.5, -0.5, -0.5), color: (1.0, 0.0, 1.0, 1.0), },
    Vertex { position: ( 0.5, -0.5,  0.5), color: (1.0, 0.0, 1.0, 1.0), },
    Vertex { position: (-0.5, -0.5, -0.5), color: (1.0, 0.0, 1.0, 1.0), },
    Vertex { position: (-0.5, -0.5,  0.5), color: (1.0, 0.0, 1.0, 1.0), },
    Vertex { position: (-0.5,  0.5, -0.5), color: (0.0, 1.0, 1.0, 1.0), },
    Vertex { position: ( 0.5,  0.5, -0.5), color: (0.0, 1.0, 1.0, 1.0), },
    Vertex { position: ( 0.5,  0.5,  0.5), color: (0.0, 1.0, 1.0, 1.0), },
    Vertex { position: ( 0.5,  0.5,  0.5), color: (0.0, 1.0, 1.0, 1.0), },
    Vertex { position: (-0.5,  0.5,  0.5), color: (0.0, 1.0, 1.0, 1.0), },
    Vertex { position: (-0.5,  0.5, -0.5), color: (0.0, 1.0, 1.0, 1.0), },
];

fn main() {
    let positions = generate_positions();
    let mut ca = setup_ca();

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
    let surface = winit::WindowBuilder::new()
        .build_vk_surface(&events_loop, instance.clone())
        .unwrap();

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

    let mut proj = cgmath::perspective(
        cgmath::Rad(std::f32::consts::FRAC_PI_2),
        { dimensions[0] as f32 / dimensions[1] as f32 },
        0.01,
        100.0,
    );
    let view = cgmath::Matrix4::look_at(
        cgmath::Point3::new(0.3, 0.3, 20.0),
        cgmath::Point3::new(0.0, 0.0, 0.0),
        cgmath::Vector3::new(0.0, 1.0, 0.0),
    );
    let scale = cgmath::Matrix4::from_scale(1.0);

    let mut vertex_buffer = update_vbuf(&ca.cells, &positions, device.clone());

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

    let mut frame_count = 0;

    loop {
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

            proj = cgmath::perspective(
                cgmath::Rad(std::f32::consts::FRAC_PI_2),
                { dimensions[0] as f32 / dimensions[1] as f32 },
                0.01,
                100.0,
            );

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

        let uniform_buffer_subbuffer = {
            let elapsed = rotation_start.elapsed();
            let rotation =
                elapsed.as_secs() as f64 + elapsed.subsec_nanos() as f64 / 1_000_000_000.0;
            let rotation = cgmath::Matrix3::from_angle_y(cgmath::Rad(rotation as f32));

            let uniform_data = vs::ty::Data {
                world: cgmath::Matrix4::from(rotation).into(),
                view: (view * scale).into(),
                proj: proj.into(),
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
            .draw(
                pipeline.clone(),
                &dynamic_state,
                vertex_buffer.clone(),
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
        events_loop.poll_events(|ev| match ev {
            winit::Event::WindowEvent {
                event: winit::WindowEvent::CloseRequested,
                ..
            } => done = true,
            winit::Event::WindowEvent {
                event:
                    winit::WindowEvent::KeyboardInput {
                        input:
                            winit::KeyboardInput {
                                virtual_keycode: Some(winit::VirtualKeyCode::N),
                                ..
                            },
                        ..
                    },
                ..
            } => {
                ca.next_gen();
                vertex_buffer = update_vbuf(&ca.cells, &positions, device.clone());
            }
            _ => (),
        });

        if done {
            break;
        }
    }

    println!("Frames: {}", frame_count);
}

fn generate_mesh(cells: &[u8], positions: &[(f32, f32, f32)]) -> Vec<Vertex> {
    cells
        .iter()
        .enumerate()
        .filter_map(|e| {
            let idx = e.0;
            let cell = e.1;

            if *cell > 0 {
                // check whether it's visible or not
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
                if count == 26 {
                    None
                } else {
                    let offset = positions[idx];
                    Some(
                        CUBE_VERTICES
                            .iter()
                            .map(move |v| {
                                let pos = v.position;
                                let color = v.color;
                                Vertex {
                                    position: (
                                        pos.0 + offset.0,
                                        pos.1 + offset.1,
                                        pos.2 + offset.2,
                                    ),
                                    color,
                                }
                            })
                            .collect::<Vec<_>>(),
                    )
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
) -> std::sync::Arc<vulkano::buffer::cpu_access::CpuAccessibleBuffer<[Vertex]>> {
    let vertices = generate_mesh(cells, positions);
    vulkano::buffer::cpu_access::CpuAccessibleBuffer::from_iter(
        device.clone(),
        vulkano::buffer::BufferUsage::vertex_buffer(),
        vertices.iter().cloned(),
    )
    .expect("failed to create buffer")
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
