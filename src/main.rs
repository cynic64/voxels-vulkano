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
use winit::{
    ElementState, Event, EventsLoop, KeyboardInput, VirtualKeyCode, WindowBuilder, WindowEvent,
};

// my stuff
mod ca;
mod camera;
mod mesher;
mod sector;

use self::mesher::Vertex;

use vulkano::sync::GpuFuture;
use vulkano_win::VkSurfaceBuild;

use std::sync::Arc;

const SIZE: usize = 256;
const SECTOR_SIDE_LEN: usize = 32;

impl_vertex!(Vertex, position, color, normal);

fn main() {
    let positions = mesher::generate_positions();
    let ca = setup_ca();
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

    let mut events_loop = EventsLoop::new();
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
    let model = glm::scale(&glm::Mat4::identity(), &glm::vec3(1.0, 1.0, 1.0));
    let mut view: [[f32; 4]; 4];
    let mut projection = glm::perspective(
        // aspect ratio
        16. / 9.,
        // fov
        1.0,
        // near
        0.1,
        // far
        100_000_000.,
    );

    let meshes = mesher::get_chunked_vertex_buffers(&ca.cells, &positions, &device.clone());
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
        w: bool,
        a: bool,
        s: bool,
        d: bool,
    }
    let mut keys_pressed = KeysPressed {
        w: false,
        a: false,
        s: false,
        d: false,
    };
    let mut last_frame = std::time::Instant::now();
    let first_frame = std::time::Instant::now();
    let mut visible_meshes;

    loop {
        let delta = get_elapsed(last_frame);
        last_frame = std::time::Instant::now();

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

            projection = glm::perspective(
                // aspect ratio
                dimensions[0] as f32 / dimensions[1] as f32,
                // fov
                1.5,
                // near
                0.1,
                // far
                100_000_000.,
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

        // camera
        if keys_pressed.w {
            cam.move_forward(delta);
        }
        if keys_pressed.s {
            cam.move_backward(delta);
        }
        if keys_pressed.a {
            cam.move_left(delta);
        }
        if keys_pressed.d {
            cam.move_right(delta);
        }

        if frame_count % 100 == 0 {
            cam.print_position();
        }

        view = cam.get_view_matrix().into();
        visible_meshes = sector::get_near_mesh_indices(&cam.position);

        let uniform_buffer_subbuffer = {
            let uniform_data = vs::ty::Data {
                world: model.into(),
                view,
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

        // building the command buffer!
        let mut command_buffer_incomplete =
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
            .unwrap();

        for &idx in visible_meshes.iter() {
            command_buffer_incomplete = command_buffer_incomplete
                .draw(
                    pipeline.clone(),
                    &dynamic_state,
                    meshes[idx].clone(),
                    set.clone(),
                    (),
                )
                .unwrap();
        }
        let command_buffer = command_buffer_incomplete
            .end_render_pass()
            .unwrap()
            .build()
            .unwrap();
        // done

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
            if let Event::WindowEvent { event, .. } = event {
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
                    WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                virtual_keycode: Some(VirtualKeyCode::N),
                                state: ElementState::Pressed,
                                ..
                            },
                        ..
                    } => {
                        println!("Implement meeeeeeee!");
                        // ca.next_gen();
                        // vertex_buffer = mesher::update_vbuf(&ca.cells, &positions, &device.clone());
                    }
                    WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                virtual_keycode: Some(VirtualKeyCode::W),
                                state: ElementState::Pressed,
                                ..
                            },
                        ..
                    } => {
                        keys_pressed.w = true;
                    }
                    WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                virtual_keycode: Some(VirtualKeyCode::A),
                                state: ElementState::Pressed,
                                ..
                            },
                        ..
                    } => {
                        keys_pressed.a = true;
                    }
                    WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                virtual_keycode: Some(VirtualKeyCode::S),
                                state: ElementState::Pressed,
                                ..
                            },
                        ..
                    } => {
                        keys_pressed.s = true;
                    }
                    WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                virtual_keycode: Some(VirtualKeyCode::D),
                                state: ElementState::Pressed,
                                ..
                            },
                        ..
                    } => {
                        keys_pressed.d = true;
                    }
                    WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                virtual_keycode: Some(VirtualKeyCode::W),
                                state: ElementState::Released,
                                ..
                            },
                        ..
                    } => {
                        keys_pressed.w = false;
                    }
                    WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                virtual_keycode: Some(VirtualKeyCode::A),
                                state: ElementState::Released,
                                ..
                            },
                        ..
                    } => {
                        keys_pressed.a = false;
                    }
                    WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                virtual_keycode: Some(VirtualKeyCode::S),
                                state: ElementState::Released,
                                ..
                            },
                        ..
                    } => {
                        keys_pressed.s = false;
                    }
                    WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                virtual_keycode: Some(VirtualKeyCode::D),
                                state: ElementState::Released,
                                ..
                            },
                        ..
                    } => {
                        keys_pressed.d = false;
                    }

                    WindowEvent::CursorMoved { position: p, .. } => {
                        let (x_diff, y_diff) = (
                            p.x - (dimensions[0] as f64 / 2.0),
                            (p.y - dimensions[1] as f64 / 2.0),
                        );
                        cam.mouse_move(x_diff as f32, y_diff as f32);
                        window
                            .set_cursor_position(winit::dpi::LogicalPosition {
                                x: dimensions[0] as f64 / 2.0,
                                y: dimensions[1] as f64 / 2.0,
                            })
                            .expect("Couldn't re-set cursor position!");
                    }
                    _ => {}
                }
            }
        });

        if done {
            break;
        }

        frame_count += 1;
    }

    let elapsed = get_elapsed(first_frame);
    let fps = (frame_count as f32) / elapsed;
    println!("FPS: {}", fps);
    println!("last delta: {}", get_elapsed(last_frame));
}

pub fn get_elapsed(start: std::time::Instant) -> f32 {
    start.elapsed().as_secs() as f32 + start.elapsed().subsec_millis() as f32 / 1000.0
}

fn setup_ca() -> ca::CellA {
    let mut ca = ca::CellA::new(SIZE, 13, 26, 14, 26);
    ca.randomize();
    for _ in 0..20 {
        ca.next_gen()
    }

    ca
}

mod vs {
    #[derive(VulkanoShader)]
    #[ty = "vertex"]
    #[src = "
#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec4 color;
layout(location = 2) in vec3 normal;

layout(location = 0) out vec4 v_color;
layout(location = 1) out vec3 v_normal;

layout(set = 0, binding = 0) uniform Data {
    mat4 world;
    mat4 view;
    mat4 proj;
} uniforms;

void main() {
    v_color = color;
    v_normal = normal;
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
layout(location = 1) in vec3 v_normal;

layout(location = 0) out vec4 f_color;

const vec3 LIGHT = vec3(3.0, 1.0, 1.0);

void main() {
    float brightness = max(dot(normalize(v_normal), normalize(LIGHT)), 0.1);
    f_color = v_color * brightness;
}
"]
    #[allow(dead_code)]
    struct Dummy;
}
