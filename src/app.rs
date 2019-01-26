extern crate nalgebra_glm as glm;
extern crate winit;

use na::{Isometry3, Point3, Vector3, Translation3, Rotation3, UnitQuaternion};
use ncollide3d::shape::Cuboid;
use ncollide3d::query::{Ray, RayCast};

extern crate vulkano_win;
use winit::{Event, EventsLoop, KeyboardInput, VirtualKeyCode, WindowBuilder, WindowEvent};

use vulkano::sync::GpuFuture;
use vulkano_win::VkSurfaceBuild;

use std::sync::Arc;

mod chunk;
mod camera;

const SIZE: u32 = 64;

#[derive(Copy, Clone, Debug)]
pub struct Vertex {
    position: (f32, f32, f32),
    color: (f32, f32, f32, f32),
    normal: (f32, f32, f32),
}
impl_vertex!(Vertex, position, color, normal);

type VertexBuffer = Arc<vulkano::buffer::immutable::ImmutableBuffer<[Vertex]>>;

pub struct App {
    vk_stuff: VkStuff,
    cam: camera::Camera,
    keys_pressed: KeysPressed,
    channels: ChannelStuff,
}

struct VkStuff {
    instance: Arc<vulkano::instance::Instance>,
    physical_device_index: usize,
    events_loop: EventsLoop,
    surface: Arc<vulkano::swapchain::Surface<winit::Window>>,
    dimensions: [u32; 2],
    device: Arc<vulkano::device::Device>,
    queue: Arc<vulkano::device::Queue>,
    swapchain_caps: vulkano::swapchain::Capabilities,
    swapchain: Arc<vulkano::swapchain::Swapchain<winit::Window>>,
    recreate_swapchain: bool,
    images: Vec<Arc<vulkano::image::SwapchainImage<winit::Window>>>,
    multisampled_color: Arc<vulkano::image::attachment::AttachmentImage>,
    depth_buffer: Arc<vulkano::image::attachment::AttachmentImage<vulkano::format::D16Unorm>>,
    multisampled_depth: Arc<vulkano::image::attachment::AttachmentImage<vulkano::format::D16Unorm>>,
    model: glm::Mat4,
    view: [[f32; 4]; 4],
    projection: glm::Mat4,
    uniform_buffer: vulkano::buffer::cpu_pool::CpuBufferPool<vs::ty::Data>,
    renderpass: Arc<vulkano::framebuffer::RenderPassAbstract + Send + Sync>,
    pipeline: Arc<vulkano::pipeline::GraphicsPipelineAbstract + Send + Sync>,
    framebuffers: Vec<Arc<vulkano::framebuffer::FramebufferAbstract + Send + Sync>>,
    dynamic_state: vulkano::command_buffer::DynamicState,
    vertex_buffer: VertexBuffer,
    previous_frame: Option<Box<GpuFuture>>,
    delta: f32,
    frame_count: u32,
}

struct ChannelStuff {
    vbuf_recv: Option<std::sync::mpsc::Receiver<VertexBuffer>>,
}

#[derive(Clone)]
struct KeysPressed {
    w: bool,
    a: bool,
    s: bool,
    d: bool,
}

impl App {
    pub fn new() -> App {
        let cam = camera::Camera::default();

        let extensions = vulkano_win::required_extensions();
        let instance = vulkano::instance::Instance::new(None, &extensions, None)
            .expect("failed to create instance");

        let physical = vulkano::instance::PhysicalDevice::from_index(&instance, 0).unwrap();
        let physical_device_index = 0;

        println!(
            "Using device: {} (type: {:?})",
            physical.name(),
            physical.ty()
        );

        let events_loop = EventsLoop::new();
        let surface = WindowBuilder::new()
            .build_vk_surface(&events_loop, instance.clone())
            .unwrap();

        let window = surface.window();
        window.hide_cursor(true);

        let dimensions;

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

        let caps = surface
            .capabilities(physical)
            .expect("failed to get surface capabilities");

        dimensions = caps.current_extent.unwrap_or([1024, 768]);

        let (swapchain, images) = {
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

        let multisampled_color =
            vulkano::image::attachment::AttachmentImage::transient_multisampled(
                device.clone(),
                caps.current_extent.unwrap_or([1024, 768]),
                4,
                caps.supported_formats[0].0,
            )
            .unwrap();

        let depth_buffer = vulkano::image::attachment::AttachmentImage::transient(
            device.clone(),
            dimensions,
            vulkano::format::D16Unorm,
        )
        .unwrap();

        let multisampled_depth =
            vulkano::image::attachment::AttachmentImage::transient_multisampled(
                device.clone(),
                dimensions,
                4,
                vulkano::format::D16Unorm,
            )
            .unwrap();

        // mvp
        let model = glm::scale(&glm::Mat4::identity(), &glm::vec3(1.0, 1.0, 1.0));
        let view: [[f32; 4]; 4] = glm::Mat4::identity().into();
        let projection = glm::perspective(
            // aspect ratio
            16. / 9.,
            // fov
            1.0,
            // near
            0.1,
            // far
            100_000_000.,
        );

        let uniform_buffer = vulkano::buffer::cpu_pool::CpuBufferPool::<vs::ty::Data>::new(
            device.clone(),
            vulkano::buffer::BufferUsage::all(),
        );

        let vs = vs::Shader::load(device.clone()).expect("failed to create shader module");
        let fs = fs::Shader::load(device.clone()).expect("failed to create shader module");

        let renderpass = Arc::new(
            single_pass_renderpass!(device.clone(),
                attachments: {
                    multisampled_color: {
                        load: Clear,
                        store: DontCare,
                        format: swapchain.format(),
                        samples: 4,
                    },
                    resolve_color: {
                        load: Clear,
                        store: Store,
                        format: swapchain.format(),
                        samples: 1,
                    },
                    multisampled_depth: {
                        load: Clear,
                        store: DontCare,
                        format: vulkano::format::Format::D16Unorm,
                        samples: 4,
                    },
                    resolve_depth: {
                        load:    DontCare,
                        store:   Store,
                        format:  vulkano::format::Format::D16Unorm,
                        samples: 1,
                        initial_layout: ImageLayout::Undefined,
                        final_layout: ImageLayout::DepthStencilAttachmentOptimal,
                    }
                },
                pass: {
                    color: [multisampled_color],
                    depth_stencil: {multisampled_depth},
                    resolve: [resolve_color],
                }
            )
            .unwrap(),
        );

        let pipeline = Arc::new(
            vulkano::pipeline::GraphicsPipeline::start()
                .vertex_input_single_buffer::<Vertex>()
                .vertex_shader(vs.main_entry_point(), ())
                .triangle_list()
                .viewports_dynamic_scissors_irrelevant(1)
                .fragment_shader(fs.main_entry_point(), ())
                .render_pass(vulkano::framebuffer::Subpass::from(renderpass.clone(), 0).unwrap())
                .depth_stencil_simple_depth()
                .cull_mode_back()
                .build(device.clone())
                .unwrap(),
        );
        let framebuffers = Vec::new();

        let recreate_swapchain = false;

        let previous_frame = Some(Box::new(vulkano::sync::now(device.clone())) as Box<GpuFuture>);

        let dynamic_state = vulkano::command_buffer::DynamicState {
            line_width: None,
            viewports: Some(vec![vulkano::pipeline::viewport::Viewport {
                origin: [0.0, 0.0],
                dimensions: [dimensions[0] as f32, dimensions[1] as f32],
                depth_range: 0.0..1.0,
            }]),
            scissors: None,
        };

        let (vbuf, future) = vulkano::buffer::immutable::ImmutableBuffer::from_iter(
            vec![
                Vertex {
                    position: (100.0, 100.0, -100.0),
                    color: (1.0, 0.0, 0.0, 1.0),
                    normal: (1.0, 1.0, 1.0),
                },
                Vertex {
                    position: (100.0, -100.0, 100.0),
                    color: (0.0, 1.0, 0.0, 1.0),
                    normal: (-1.0, 1.0, 1.0),
                },
                Vertex {
                    position: (-100.0, -100.0, 0.5),
                    color: (0.0, 0.0, 1.0, 1.0),
                    normal: (1.0, 1.0, -1.0),
                },
            ]
            .iter()
            .cloned(),
            vulkano::buffer::BufferUsage::vertex_buffer(),
            queue.clone(),
        )
        .expect("failed to create buffer");
        future.flush().unwrap();

        let keys_pressed = KeysPressed {
            w: false,
            a: false,
            s: false,
            d: false,
        };

        let delta = 0.0;

        let channels = ChannelStuff { vbuf_recv: None };

        App {
            vk_stuff: VkStuff {
                previous_frame,
                instance,
                physical_device_index,
                events_loop,
                surface,
                dimensions,
                device,
                queue,
                swapchain_caps: caps,
                swapchain,
                recreate_swapchain,
                images,
                multisampled_color,
                depth_buffer,
                multisampled_depth,
                model,
                view,
                projection,
                uniform_buffer,
                renderpass,
                pipeline,
                framebuffers,
                dynamic_state,
                vertex_buffer: vbuf,
                delta,
                frame_count: 0,
            },
            cam,
            keys_pressed,
            channels,
        }
    }

    pub fn run(&mut self) {
        self.channels.vbuf_recv = Some(Self::spawn_thread(self.vk_stuff.queue.clone()));

        let start = std::time::Instant::now();
        loop {
            let start = std::time::Instant::now();
            let done = self.draw_frame();
            self.vk_stuff.delta = get_elapsed(start);
            if done {
                break;
            }
        }

        let fps = (self.vk_stuff.frame_count as f32) / get_elapsed(start);
        println!("Average FPS: {}", fps);
    }

    fn spawn_thread(queue: Arc<vulkano::device::Queue>) -> std::sync::mpsc::Receiver<VertexBuffer> {
        let (vbuf_trans, vbuf_recv) = std::sync::mpsc::channel();
        let mut ch = chunk::Chunk::new(queue.clone());
        ch.update_positions();

        std::thread::spawn(move || loop {
            ch.randomize_state();
            ch.update_vbuf(queue.clone());
            let vbuf = ch.get_vbuf();
            vbuf_trans.send(vbuf).unwrap();

            std::thread::sleep(std::time::Duration::from_millis(2000));
        });

        vbuf_recv
    }

    fn draw_frame(&mut self) -> bool {
        // previous frame magic
        self.vk_stuff
            .previous_frame
            .as_mut()
            .unwrap()
            .cleanup_finished();

        // check for updates from the thread
        self.check_channels();

        // rebuild stuff if we need to
        if self.vk_stuff.recreate_swapchain {
            self.rebuild_swapchain();
        }

        if self.vk_stuff.framebuffers.is_empty() {
            self.rebuild_framebuffers();
        }

        // get access to a swapchain image
        let (image_num, acquire_future) =
            match vulkano::swapchain::acquire_next_image(self.vk_stuff.swapchain.clone(), None) {
                Ok(r) => r,
                Err(vulkano::swapchain::AcquireError::OutOfDate) => {
                    self.vk_stuff.recreate_swapchain = true;
                    return false;
                }
                Err(err) => panic!("{:?}", err),
            };

        // build the command buffer
        let command_buffer = self.build_command_buffer(image_num);

        // send commands to GPU
        let future = self
            .vk_stuff
            .previous_frame
            .take()
            .unwrap()
            .join(acquire_future)
            .then_execute(self.vk_stuff.queue.clone(), command_buffer)
            .unwrap()
            .then_swapchain_present(
                self.vk_stuff.queue.clone(),
                self.vk_stuff.swapchain.clone(),
                image_num,
            )
            .then_signal_fence_and_flush();

        // see if the commands were executed OK
        match future {
            Ok(future) => {
                self.vk_stuff.previous_frame = Some(Box::new(future));
            }
            Err(vulkano::sync::FlushError::OutOfDate) => {
                self.vk_stuff.recreate_swapchain = true;
                self.vk_stuff.previous_frame =
                    Some(Box::new(vulkano::sync::now(self.vk_stuff.device.clone())) as Box<_>);
            }
            Err(e) => {
                println!("{:?}", e);
                self.vk_stuff.previous_frame =
                    Some(Box::new(vulkano::sync::now(self.vk_stuff.device.clone())) as Box<_>);
            }
        }

        self.update_camera();
        self.vk_stuff.frame_count += 1;

        self.poll_events()
    }

    fn check_channels(&mut self) {
        if self.channels.recv.is_some() {
            let result = self.channels.recv.as_mut().unwrap().try_recv();
            if result.is_ok() {
                println!("Got a new vertex buffer!");
                self.vk_stuff.vertex_buffer = result.unwrap();
            }
        } else {
            println!("Reciever uninitialized!");
        }
    }

    fn poll_events(&mut self) -> bool {
        // closures are a pain in the ass!
        // and caused this mess
        let mut done = false;
        let dimensions = self.vk_stuff.dimensions;
        let mut x_movement = 0.0;
        let mut y_movement = 0.0;

        let mut keys_pressed = self.keys_pressed.clone();

        self.vk_stuff.events_loop.poll_events(|event| {
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

                    // Mouse movement
                    WindowEvent::CursorMoved { position: p, .. } => {
                        let (x_diff, y_diff) = (
                            p.x - (dimensions[0] as f64 / 2.0),
                            p.y - (dimensions[1] as f64 / 2.0),
                        );
                        x_movement = x_diff as f32;
                        y_movement = y_diff as f32;
                    }
                    // WASD down
                    WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                virtual_keycode: Some(VirtualKeyCode::W),
                                state: winit::ElementState::Pressed,
                                ..
                            },
                        ..
                    } => keys_pressed.w = true,
                    WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                virtual_keycode: Some(VirtualKeyCode::A),
                                state: winit::ElementState::Pressed,
                                ..
                            },
                        ..
                    } => keys_pressed.a = true,
                    WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                virtual_keycode: Some(VirtualKeyCode::S),
                                state: winit::ElementState::Pressed,
                                ..
                            },
                        ..
                    } => keys_pressed.s = true,
                    WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                virtual_keycode: Some(VirtualKeyCode::D),
                                state: winit::ElementState::Pressed,
                                ..
                            },
                        ..
                    } => keys_pressed.d = true,
                    // WASD up
                    WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                virtual_keycode: Some(VirtualKeyCode::W),
                                state: winit::ElementState::Released,
                                ..
                            },
                        ..
                    } => keys_pressed.w = false,
                    WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                virtual_keycode: Some(VirtualKeyCode::A),
                                state: winit::ElementState::Released,
                                ..
                            },
                        ..
                    } => keys_pressed.a = false,
                    WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                virtual_keycode: Some(VirtualKeyCode::S),
                                state: winit::ElementState::Released,
                                ..
                            },
                        ..
                    } => keys_pressed.s = false,
                    WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                virtual_keycode: Some(VirtualKeyCode::D),
                                state: winit::ElementState::Released,
                                ..
                            },
                        ..
                    } => keys_pressed.d = false,
                    _ => {}
                }
            }
        });

        // reset cursor and change camera view
        self.vk_stuff
            .surface
            .window()
            .set_cursor_position(winit::dpi::LogicalPosition {
                x: self.vk_stuff.dimensions[0] as f64 / 2.0,
                y: self.vk_stuff.dimensions[1] as f64 / 2.0,
            })
            .expect("Couldn't re-set cursor position!");
        self.cam.mouse_move(x_movement as f32, y_movement as f32);

        // Ray - for later...
        // let orig = self.cam.position;
        // let dir = self.cam.front;
        // let cuboid = Cuboid::new(Vector3::new(16.0, 16.0, 16.0));
        // let ray = Ray::new(orig.into(), dir);

        // let isom = Isometry3::from_parts(
        //     Translation3::new(16.0, 16.0, 16.0),
        //     UnitQuaternion::from_scaled_axis(Vector3::y() * 0.0)
        // );
        // if cuboid.toi_with_ray(&isom, &ray, true).is_some() {
        //     println!("Collision!, {}", rand::random::<u8>());
        // }

        // update keys_pressed
        self.keys_pressed = keys_pressed;

        done
    }

    fn rebuild_swapchain(&mut self) {
        let physical = vulkano::instance::PhysicalDevice::from_index(
            &self.vk_stuff.instance,
            self.vk_stuff.physical_device_index,
        )
        .unwrap();

        self.vk_stuff.dimensions = self
            .vk_stuff
            .surface
            .capabilities(physical)
            .expect("failed to get surface capabilities")
            .current_extent
            .unwrap_or([1024, 768]);

        self.vk_stuff.multisampled_color =
            vulkano::image::attachment::AttachmentImage::transient_multisampled(
                self.vk_stuff.device.clone(),
                self.vk_stuff.dimensions,
                4,
                self.vk_stuff.swapchain_caps.supported_formats[0].0,
            )
            .unwrap();

        self.vk_stuff.multisampled_depth =
            vulkano::image::attachment::AttachmentImage::transient_multisampled(
                self.vk_stuff.device.clone(),
                self.vk_stuff.dimensions,
                4,
                vulkano::format::D16Unorm,
            )
            .unwrap();

        let (new_swapchain, new_images) = match self
            .vk_stuff
            .swapchain
            .recreate_with_dimension(self.vk_stuff.dimensions)
        {
            Ok(r) => r,
            Err(vulkano::swapchain::SwapchainCreationError::UnsupportedDimensions) => {
                return;
            }
            Err(err) => panic!("{:?}", err),
        };

        self.vk_stuff.swapchain = new_swapchain;
        self.vk_stuff.images = new_images;

        self.vk_stuff.depth_buffer = vulkano::image::attachment::AttachmentImage::transient(
            self.vk_stuff.device.clone(),
            self.vk_stuff.dimensions,
            vulkano::format::D16Unorm,
        )
        .unwrap();

        self.vk_stuff.framebuffers = Vec::new();

        self.vk_stuff.projection = glm::perspective(
            // aspect ratio
            self.vk_stuff.dimensions[0] as f32 / self.vk_stuff.dimensions[1] as f32,
            // fov
            1.5,
            // near
            0.1,
            // far
            100_000_000.,
        );

        self.vk_stuff.dynamic_state.viewports = Some(vec![vulkano::pipeline::viewport::Viewport {
            origin: [0.0, 0.0],
            dimensions: [
                self.vk_stuff.dimensions[0] as f32,
                self.vk_stuff.dimensions[1] as f32,
            ],
            depth_range: 0.0..1.0,
        }]);

        self.vk_stuff.recreate_swapchain = false;
    }

    fn rebuild_framebuffers(&mut self) {
        self.vk_stuff.framebuffers = self
            .vk_stuff
            .images
            .iter()
            .map(|image| {
                let fba: Arc<vulkano::framebuffer::FramebufferAbstract + Send + Sync> = Arc::new(
                    vulkano::framebuffer::Framebuffer::start(self.vk_stuff.renderpass.clone())
                        .add(self.vk_stuff.multisampled_color.clone())
                        .unwrap()
                        .add(image.clone())
                        .unwrap()
                        .add(self.vk_stuff.multisampled_depth.clone())
                        .unwrap()
                        .add(self.vk_stuff.depth_buffer.clone())
                        .unwrap()
                        .build()
                        .unwrap(),
                );

                fba
            })
            .collect::<Vec<_>>();
    }

    fn build_command_buffer(
        &mut self,
        image_num: usize,
    ) -> vulkano::command_buffer::AutoCommandBuffer {
        let uniform_buffer_subbuffer = {
            let uniform_data = vs::ty::Data {
                world: self.vk_stuff.model.into(),
                view: self.vk_stuff.view,
                proj: self.vk_stuff.projection.into(),
            };

            self.vk_stuff.uniform_buffer.next(uniform_data).unwrap()
        };

        // long type! :(
        let uniform_set = Arc::new(
            vulkano::descriptor::descriptor_set::PersistentDescriptorSet::start(
                self.vk_stuff.pipeline.clone(),
                0,
            )
            .add_buffer(uniform_buffer_subbuffer)
            .unwrap()
            .build()
            .unwrap(),
        );

        vulkano::command_buffer::AutoCommandBufferBuilder::primary_one_time_submit(
            self.vk_stuff.device.clone(),
            self.vk_stuff.queue.family(),
        )
        .unwrap()
        .begin_render_pass(
            self.vk_stuff.framebuffers[image_num].clone(),
            false,
            vec![
                [0.2, 0.2, 0.2, 1.0].into(),
                [0.2, 0.2, 0.2, 1.0].into(),
                1f32.into(),
                vulkano::format::ClearValue::None,
            ],
        )
        .unwrap()
        .draw(
            self.vk_stuff.pipeline.clone(),
            &self.vk_stuff.dynamic_state,
            vec![self.vk_stuff.vertex_buffer.clone()],
            uniform_set.clone(),
            (),
        )
        .unwrap()
        .end_render_pass()
        .unwrap()
        .build()
        .unwrap()
    }

    fn update_camera(&mut self) {
        // move the camera (maybe)
        if self.keys_pressed.w {
            self.cam.move_forward(self.vk_stuff.delta);
        }
        if self.keys_pressed.s {
            self.cam.move_backward(self.vk_stuff.delta);
        }
        if self.keys_pressed.a {
            self.cam.move_left(self.vk_stuff.delta);
        }
        if self.keys_pressed.d {
            self.cam.move_right(self.vk_stuff.delta);
        }

        // update our view matrix to match the camera's
        self.vk_stuff.view = self.cam.get_view_matrix().into();
    }
}

fn get_elapsed(start: std::time::Instant) -> f32 {
    start.elapsed().as_secs() as f32 + start.elapsed().subsec_nanos() as f32 / 1_000_000_000.0
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

const vec3 LIGHT = vec3(3.0, 2.0, 1.0);

void main() {
    float brightness = dot(normalize(v_normal), normalize(LIGHT));
    vec3 dark_color = v_color.xyz * 0.6;
    vec3 regular_color = v_color.xyz;

    f_color = vec4(mix(dark_color, regular_color, brightness), 1.0);
}
"]
    #[allow(dead_code)]
    struct Dummy;
}
