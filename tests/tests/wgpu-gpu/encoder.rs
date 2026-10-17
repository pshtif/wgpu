use wgpu::util::DeviceExt;
use wgpu::CommandEncoder;
use wgpu_test::{
    fail, gpu_test, FailureCase, GpuTestConfiguration, GpuTestInitializer, TestParameters,
    TestingContext,
};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.extend([
        DROP_ENCODER,
        DROP_QUEUE_BEFORE_CREATING_COMMAND_ENCODER,
        DROP_ENCODER_AFTER_ERROR,
        ENCODER_OPERATIONS_FAIL_WHILE_PASS_ALIVE,
        MANY_PASSES_IN_ONE_ENCODER,
    ]);
}

#[gpu_test]
static DROP_ENCODER: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().enable_noop())
    .run_sync(|ctx| {
        let encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        drop(encoder);
    });

#[gpu_test]
static DROP_QUEUE_BEFORE_CREATING_COMMAND_ENCODER: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(
            TestParameters::default()
                .expect_fail(FailureCase::always())
                .enable_noop(),
        )
        .run_sync(|ctx| {
            // Use the device after the queue is dropped. Currently this panics
            // but it probably shouldn't.
            // TODO(https://github.com/gfx-rs/wgpu/issues/7781) revisit this
            let TestingContext { device, queue, .. } = ctx;
            drop(queue);
            let _encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        });

#[gpu_test]
static DROP_ENCODER_AFTER_ERROR: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().enable_noop())
    .run_sync(|ctx| {
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        let target_tex = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: 100,
                height: 100,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let target_view = target_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let mut renderpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("renderpass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                ops: wgpu::Operations::default(),
                resolve_target: None,
                view: &target_view,
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        // This viewport is invalid because it has negative size.
        renderpass.set_viewport(0.0, 0.0, -1.0, -1.0, 0.0, 1.0);
        drop(renderpass);

        fail(&ctx.device, || encoder.finish(), Some("less than zero"));
    });

#[gpu_test]
static ENCODER_OPERATIONS_FAIL_WHILE_PASS_ALIVE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(wgpu::Features::CLEAR_TEXTURE | wgpu::Features::TIMESTAMP_QUERY)
            .enable_noop(),
    )
    .run_sync(encoder_operations_fail_while_pass_alive);

fn encoder_operations_fail_while_pass_alive(ctx: TestingContext) {
    let buffer_source = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: &[0u8; 4],
            usage: wgpu::BufferUsages::COPY_SRC,
        });
    let buffer_dest = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: &[0u8; 4],
            usage: wgpu::BufferUsages::COPY_DST,
        });

    let texture_desc = wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    };
    let texture_dst = ctx.device.create_texture(&texture_desc);
    let texture_src = ctx.device.create_texture(&wgpu::TextureDescriptor {
        usage: wgpu::TextureUsages::COPY_SRC,
        ..texture_desc
    });
    let query_set = ctx.device.create_query_set(&wgpu::QuerySetDescriptor {
        count: 1,
        ty: wgpu::QueryType::Timestamp,
        label: None,
    });

    let target_desc = wgpu::TextureDescriptor {
        label: Some("target_tex"),
        size: wgpu::Extent3d {
            width: 4,
            height: 4,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Bgra8UnormSrgb,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[wgpu::TextureFormat::Bgra8UnormSrgb],
    };
    let target_tex = ctx.device.create_texture(&target_desc);
    let color_attachment_view = target_tex.create_view(&wgpu::TextureViewDescriptor::default());

    #[allow(clippy::type_complexity)]
    let recording_ops: Vec<(_, Box<dyn Fn(&mut CommandEncoder)>)> = vec![
        (
            "begin_compute_pass",
            Box::new(|encoder: &mut wgpu::CommandEncoder| {
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            }),
        ),
        (
            "begin_render_pass",
            Box::new(|encoder: &mut wgpu::CommandEncoder| {
                encoder.begin_render_pass(&wgpu::RenderPassDescriptor::default());
            }),
        ),
        (
            "copy_buffer_to_buffer",
            Box::new(|encoder: &mut wgpu::CommandEncoder| {
                encoder.copy_buffer_to_buffer(&buffer_source, 0, &buffer_dest, 0, 4);
            }),
        ),
        (
            "copy_buffer_to_texture",
            Box::new(|encoder: &mut wgpu::CommandEncoder| {
                encoder.copy_buffer_to_texture(
                    wgpu::TexelCopyBufferInfo {
                        buffer: &buffer_source,
                        layout: wgpu::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(4),
                            rows_per_image: None,
                        },
                    },
                    texture_dst.as_image_copy(),
                    texture_dst.size(),
                );
            }),
        ),
        (
            "copy_texture_to_buffer",
            Box::new(|encoder: &mut wgpu::CommandEncoder| {
                encoder.copy_texture_to_buffer(
                    wgpu::TexelCopyTextureInfo {
                        texture: &texture_src,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::TexelCopyBufferInfo {
                        buffer: &buffer_dest,
                        layout: wgpu::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(4),
                            rows_per_image: None,
                        },
                    },
                    texture_dst.size(),
                );
            }),
        ),
        (
            "copy_texture_to_texture",
            Box::new(|encoder: &mut wgpu::CommandEncoder| {
                encoder.copy_texture_to_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &texture_src,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::TexelCopyTextureInfo {
                        texture: &texture_dst,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    texture_dst.size(),
                );
            }),
        ),
        (
            "clear_texture",
            Box::new(|encoder: &mut wgpu::CommandEncoder| {
                encoder.clear_texture(&texture_dst, &wgpu::ImageSubresourceRange::default());
            }),
        ),
        (
            "clear_buffer",
            Box::new(|encoder: &mut wgpu::CommandEncoder| {
                encoder.clear_buffer(&buffer_dest, 0, None);
            }),
        ),
        (
            "insert_debug_marker",
            Box::new(|encoder: &mut wgpu::CommandEncoder| {
                encoder.insert_debug_marker("marker");
            }),
        ),
        (
            "push_debug_group",
            Box::new(|encoder: &mut wgpu::CommandEncoder| {
                encoder.push_debug_group("marker");
            }),
        ),
        (
            "pop_debug_group",
            Box::new(|encoder: &mut wgpu::CommandEncoder| {
                encoder.pop_debug_group();
            }),
        ),
        (
            "resolve_query_set",
            Box::new(|encoder: &mut wgpu::CommandEncoder| {
                encoder.resolve_query_set(&query_set, 0..1, &buffer_dest, 0);
            }),
        ),
        (
            "write_timestamp",
            Box::new(|encoder: &mut wgpu::CommandEncoder| {
                encoder.write_timestamp(&query_set, 0);
            }),
        ),
    ];

    #[derive(Clone, Copy, Debug)]
    enum PassType {
        Compute,
        Render,
    }

    let create_pass = |encoder: &mut wgpu::CommandEncoder, pass_type| -> Box<dyn std::any::Any> {
        match pass_type {
            PassType::Compute => Box::new(
                encoder
                    .begin_compute_pass(&wgpu::ComputePassDescriptor::default())
                    .forget_lifetime(),
            ),
            PassType::Render => Box::new(
                encoder
                    .begin_render_pass(&wgpu::RenderPassDescriptor {
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &color_attachment_view,
                            depth_slice: None,
                            resolve_target: None,
                            ops: wgpu::Operations::default(),
                        })],
                        ..Default::default()
                    })
                    .forget_lifetime(),
            ),
        }
    };

    for &pass_type in [PassType::Compute, PassType::Render].iter() {
        for (op_name, op) in recording_ops.iter() {
            // Test the case where the pass is not ended before calling finish()
            let mut encoder = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

            let pass = create_pass(&mut encoder, pass_type);

            let _scope = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);

            log::info!("Testing operation {op_name:?} on a locked command encoder while a {pass_type:?} pass is active");
            op(&mut encoder);

            fail(&ctx.device, || encoder.finish(), Some("encoder is locked"));

            drop(pass);

            // ...and the case where the pass is ended before calling finish()
            let mut encoder = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

            let pass = create_pass(&mut encoder, pass_type);

            log::info!("Testing operation {op_name:?} on a locked command encoder while a {pass_type:?} pass is active");
            op(&mut encoder);

            drop(pass);

            fail(&ctx.device, || encoder.finish(), Some("encoder is locked"));

            // We don't care about any errors that happen outside of a `fail` call.
            drop(_scope);
        }
    }
}

/// More passes than Metal allows command buffers to be outstanding, each
/// depending on the one before it.
#[gpu_test]
static MANY_PASSES_IN_ONE_ENCODER: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().test_features_limits())
    .run_async(|ctx| async move {
        const ITERATIONS: u32 = 2100;

        let compute_module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(
                    r#"
                    @group(0) @binding(0) var<storage, read_write> counter: u32;

                    @compute @workgroup_size(1)
                    fn main() {
                        counter += 1u;
                    }
                    "#
                    .into(),
                ),
            });
        let render_module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(
                    r#"
                    @group(0) @binding(0) var<storage, read> counter: u32;

                    @vertex
                    fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
                        let uv = vec2f(f32((index << 1u) & 2u), f32(index & 2u));
                        return vec4f(uv * 2.0 - 1.0, 0.0, 1.0);
                    }

                    @fragment
                    fn fs_main() -> @location(0) u32 {
                        return counter;
                    }
                    "#
                    .into(),
                ),
            });

        let counter = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let target = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Uint,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());
        let readback = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 512,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let compute_pipeline =
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: None,
                    layout: None,
                    module: &compute_module,
                    entry_point: None,
                    compilation_options: Default::default(),
                    cache: None,
                });
        let render_pipeline = ctx
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None,
                layout: None,
                vertex: wgpu::VertexState {
                    module: &render_module,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &render_module,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::TextureFormat::R32Uint.into())],
                }),
                multiview_mask: None,
                cache: None,
            });
        let compute_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &compute_pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: counter.as_entire_binding(),
            }],
        });
        let render_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &render_pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: counter.as_entire_binding(),
            }],
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.clear_buffer(&counter, 0, None);
        for _ in 0..ITERATIONS {
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                pass.set_pipeline(&compute_pipeline);
                pass.set_bind_group(0, &compute_bind_group, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations::default(),
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&render_pipeline);
            pass.set_bind_group(0, &render_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        encoder.copy_buffer_to_buffer(&counter, 0, &readback, 0, 4);
        encoder.copy_texture_to_buffer(
            target.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 256,
                    bytes_per_row: None,
                    rows_per_image: None,
                },
            },
            target.size(),
        );
        ctx.queue.submit([encoder.finish()]);

        readback.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        ctx.async_poll(wgpu::PollType::wait_indefinitely())
            .await
            .unwrap();
        let data = readback.slice(..).get_mapped_range().unwrap();
        let counter_value: u32 = *bytemuck::from_bytes(&data[0..4]);
        let target_value: u32 = *bytemuck::from_bytes(&data[256..260]);
        assert_eq!(counter_value, ITERATIONS);
        assert_eq!(target_value, ITERATIONS);
    });
