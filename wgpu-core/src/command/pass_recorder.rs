//! A pass's barriers, discarded surface initialization, query resets and
//! indirect validation must be encoded before the pass, but are only known
//! once it has been validated. Passes record their HAL commands while being
//! validated, so that both can be encoded into the same HAL command buffer.

use alloc::{boxed::Box, string::String, sync::Arc, vec::Vec};
use core::{num::NonZeroU32, ops::Range};

use wgt::DynamicOffset;

use crate::{
    binding_model::{BindGroup, PipelineLayout},
    command::{bundle::RenderBundle, DrawCommandFamily},
    device::{Device, DeviceError},
    indirect_validation::DrawResources,
    pipeline::{ComputePipeline, RenderPipeline},
    resource::{
        Buffer, DestroyedResourceError, InvalidOrDestroyedResourceError, Labeled, QuerySet,
        RawResourceAccess, Texture, TextureView,
    },
    snatch::SnatchGuard,
    track::PendingTransition,
};

/// A buffer used by a recorded command.
#[derive(Debug)]
pub(crate) enum RecordedBuffer {
    /// A user buffer.
    Buffer(Arc<Buffer>),
    /// The destination buffer of indirect draw validation with this index in
    /// [`DrawResources`].
    DrawValidation(usize),
    /// The destination buffer of indirect dispatch validation.
    DispatchValidation,
}

/// The family of a recorded indirect draw.
#[derive(Clone, Copy, Debug)]
pub(crate) enum IndirectDrawKind {
    Draw,
    DrawIndexed,
    DrawMeshTasks,
}

impl From<DrawCommandFamily> for IndirectDrawKind {
    fn from(family: DrawCommandFamily) -> Self {
        match family {
            DrawCommandFamily::Draw => Self::Draw,
            DrawCommandFamily::DrawIndexed => Self::DrawIndexed,
            DrawCommandFamily::DrawMeshTasks => Self::DrawMeshTasks,
        }
    }
}

#[derive(Debug)]
pub(crate) struct RecordedAttachment {
    pub view: Arc<TextureView>,
    pub usage: wgt::TextureUses,
}

#[derive(Debug)]
pub(crate) struct RecordedColorAttachment {
    pub target: RecordedAttachment,
    pub depth_slice: Option<u32>,
    pub resolve_target: Option<RecordedAttachment>,
    pub ops: hal::AttachmentOps,
    pub clear_value: wgt::Color,
}

#[derive(Debug)]
pub(crate) struct RecordedDepthStencilAttachment {
    pub target: RecordedAttachment,
    pub depth_ops: hal::AttachmentOps,
    pub stencil_ops: hal::AttachmentOps,
    pub depth_read_only: bool,
    pub stencil_read_only: bool,
    pub clear_value: (f32, u32),
}

#[derive(Debug)]
pub(crate) struct RecordedTimestampWrites {
    pub query_set: Arc<QuerySet>,
    pub beginning_of_pass_write_index: Option<u32>,
    pub end_of_pass_write_index: Option<u32>,
}

/// The owned equivalent of [`hal::RenderPassDescriptor`].
#[derive(Debug)]
pub(crate) struct RecordedRenderPass {
    pub label: Option<String>,
    pub extent: wgt::Extent3d,
    pub sample_count: u32,
    pub color_attachments: Vec<Option<RecordedColorAttachment>>,
    pub depth_stencil_attachment: Option<RecordedDepthStencilAttachment>,
    pub multiview_mask: Option<NonZeroU32>,
    pub timestamp_writes: Option<RecordedTimestampWrites>,
    pub occlusion_query_set: Option<Arc<QuerySet>>,
}

#[derive(Debug)]
enum Command {
    BeginRenderPass(Box<RecordedRenderPass>),
    EndRenderPass,
    BeginComputePass {
        label: Option<String>,
        timestamp_writes: Option<RecordedTimestampWrites>,
    },
    EndComputePass,
    SetRenderPipeline(Arc<RenderPipeline>),
    SetComputePipeline(Arc<ComputePipeline>),
    SetBindGroup {
        layout: Arc<PipelineLayout>,
        index: u32,
        bind_group: Arc<BindGroup>,
        /// Range in [`PassRecorder::data`].
        dynamic_offsets: Range<usize>,
    },
    SetImmediates {
        layout: Arc<PipelineLayout>,
        offset_bytes: u32,
        /// Range in [`PassRecorder::data`].
        data: Range<usize>,
    },
    SetIndexBuffer {
        buffer: Arc<Buffer>,
        offset: wgt::BufferAddress,
        size: wgt::BufferAddress,
        format: wgt::IndexFormat,
    },
    SetVertexBuffer {
        slot: u32,
        buffer: Arc<Buffer>,
        offset: wgt::BufferAddress,
        size: wgt::BufferAddress,
    },
    SetViewport {
        rect: hal::Rect<f32>,
        depth_range: Range<f32>,
    },
    SetScissorRect(hal::Rect<u32>),
    SetStencilReference(u32),
    SetBlendConstants([f32; 4]),
    Draw {
        first_vertex: u32,
        vertex_count: u32,
        first_instance: u32,
        instance_count: u32,
    },
    DrawIndexed {
        first_index: u32,
        index_count: u32,
        base_vertex: i32,
        first_instance: u32,
        instance_count: u32,
    },
    DrawMeshTasks([u32; 3]),
    DrawIndirect {
        kind: IndirectDrawKind,
        buffer: RecordedBuffer,
        offset: wgt::BufferAddress,
        draw_count: u32,
    },
    DrawIndirectCount {
        kind: IndirectDrawKind,
        buffer: Arc<Buffer>,
        offset: wgt::BufferAddress,
        count_buffer: Arc<Buffer>,
        count_offset: wgt::BufferAddress,
        max_count: u32,
    },
    ExecuteBundle {
        bundle: Arc<RenderBundle>,
        /// The validated indirect draws of the bundle, in order:
        /// (index in [`DrawResources`], offset).
        validated_draws: Vec<(usize, wgt::BufferAddress)>,
    },
    DispatchWorkgroups([u32; 3]),
    DispatchWorkgroupsIndirect {
        buffer: RecordedBuffer,
        offset: wgt::BufferAddress,
    },
    /// Binds the indirect dispatch validation pipeline and its resources for
    /// validating `buffer` at `offset`.
    SetDispatchValidation {
        buffer: Arc<Buffer>,
        offset: wgt::BufferAddress,
    },
    TransitionBuffers(Vec<(RecordedBuffer, hal::StateTransition<wgt::BufferUses>)>),
    TransitionTextures(Vec<(Arc<Texture>, PendingTransition<wgt::TextureUses>)>),
    BeginQuery(Arc<QuerySet>, u32),
    EndQuery(Arc<QuerySet>, u32),
    WriteTimestamp(Arc<QuerySet>, u32),
    ResetQueries(Arc<QuerySet>, Range<u32>),
    /// Range in [`PassRecorder::labels`].
    BeginDebugMarker(Range<usize>),
    EndDebugMarker,
    /// Range in [`PassRecorder::labels`].
    InsertDebugMarker(Range<usize>),
}

#[derive(Debug, Default)]
pub(crate) struct PassRecorder {
    commands: Vec<Command>,
    /// Dynamic offsets and immediate data of the recorded commands.
    data: Vec<u32>,
    /// Debug labels of the recorded commands.
    labels: String,
}

impl PassRecorder {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    fn push_data(&mut self, data: &[u32]) -> Range<usize> {
        let start = self.data.len();
        self.data.extend_from_slice(data);
        start..self.data.len()
    }

    fn push_label(&mut self, label: &str) -> Range<usize> {
        let start = self.labels.len();
        self.labels.push_str(label);
        start..self.labels.len()
    }

    pub(crate) fn begin_render_pass(&mut self, desc: RecordedRenderPass) {
        self.commands.push(Command::BeginRenderPass(Box::new(desc)));
    }

    pub(crate) fn end_render_pass(&mut self) {
        self.commands.push(Command::EndRenderPass);
    }

    pub(crate) fn begin_compute_pass(
        &mut self,
        label: Option<String>,
        timestamp_writes: Option<RecordedTimestampWrites>,
    ) {
        self.commands.push(Command::BeginComputePass {
            label,
            timestamp_writes,
        });
    }

    pub(crate) fn end_compute_pass(&mut self) {
        self.commands.push(Command::EndComputePass);
    }

    pub(crate) fn set_render_pipeline(&mut self, pipeline: &Arc<RenderPipeline>) {
        self.commands
            .push(Command::SetRenderPipeline(pipeline.clone()));
    }

    pub(crate) fn set_compute_pipeline(&mut self, pipeline: &Arc<ComputePipeline>) {
        self.commands
            .push(Command::SetComputePipeline(pipeline.clone()));
    }

    pub(crate) fn set_bind_group(
        &mut self,
        layout: &Arc<PipelineLayout>,
        index: u32,
        bind_group: &Arc<BindGroup>,
        dynamic_offsets: &[DynamicOffset],
    ) {
        let dynamic_offsets = self.push_data(dynamic_offsets);
        self.commands.push(Command::SetBindGroup {
            layout: layout.clone(),
            index,
            bind_group: bind_group.clone(),
            dynamic_offsets,
        });
    }

    pub(crate) fn set_immediates(
        &mut self,
        layout: &Arc<PipelineLayout>,
        offset_bytes: u32,
        data: &[u32],
    ) {
        let data = self.push_data(data);
        self.commands.push(Command::SetImmediates {
            layout: layout.clone(),
            offset_bytes,
            data,
        });
    }

    /// The binding must have been validated.
    pub(crate) fn set_index_buffer(
        &mut self,
        buffer: &Arc<Buffer>,
        offset: wgt::BufferAddress,
        size: wgt::BufferAddress,
        format: wgt::IndexFormat,
    ) {
        self.commands.push(Command::SetIndexBuffer {
            buffer: buffer.clone(),
            offset,
            size,
            format,
        });
    }

    /// The binding must have been validated.
    pub(crate) fn set_vertex_buffer(
        &mut self,
        slot: u32,
        buffer: &Arc<Buffer>,
        offset: wgt::BufferAddress,
        size: wgt::BufferAddress,
    ) {
        self.commands.push(Command::SetVertexBuffer {
            slot,
            buffer: buffer.clone(),
            offset,
            size,
        });
    }

    pub(crate) fn set_viewport(&mut self, rect: hal::Rect<f32>, depth_range: Range<f32>) {
        self.commands
            .push(Command::SetViewport { rect, depth_range });
    }

    pub(crate) fn set_scissor_rect(&mut self, rect: hal::Rect<u32>) {
        self.commands.push(Command::SetScissorRect(rect));
    }

    pub(crate) fn set_stencil_reference(&mut self, value: u32) {
        self.commands.push(Command::SetStencilReference(value));
    }

    pub(crate) fn set_blend_constants(&mut self, color: [f32; 4]) {
        self.commands.push(Command::SetBlendConstants(color));
    }

    pub(crate) fn draw(
        &mut self,
        first_vertex: u32,
        vertex_count: u32,
        first_instance: u32,
        instance_count: u32,
    ) {
        self.commands.push(Command::Draw {
            first_vertex,
            vertex_count,
            first_instance,
            instance_count,
        });
    }

    pub(crate) fn draw_indexed(
        &mut self,
        first_index: u32,
        index_count: u32,
        base_vertex: i32,
        first_instance: u32,
        instance_count: u32,
    ) {
        self.commands.push(Command::DrawIndexed {
            first_index,
            index_count,
            base_vertex,
            first_instance,
            instance_count,
        });
    }

    pub(crate) fn draw_mesh_tasks(&mut self, group_count: [u32; 3]) {
        self.commands.push(Command::DrawMeshTasks(group_count));
    }

    pub(crate) fn draw_indirect(
        &mut self,
        kind: IndirectDrawKind,
        buffer: RecordedBuffer,
        offset: wgt::BufferAddress,
        draw_count: u32,
    ) {
        self.commands.push(Command::DrawIndirect {
            kind,
            buffer,
            offset,
            draw_count,
        });
    }

    pub(crate) fn draw_indirect_count(
        &mut self,
        kind: IndirectDrawKind,
        buffer: &Arc<Buffer>,
        offset: wgt::BufferAddress,
        count_buffer: &Arc<Buffer>,
        count_offset: wgt::BufferAddress,
        max_count: u32,
    ) {
        self.commands.push(Command::DrawIndirectCount {
            kind,
            buffer: buffer.clone(),
            offset,
            count_buffer: count_buffer.clone(),
            count_offset,
            max_count,
        });
    }

    /// `validated_draws` come from [`RenderBundle::validate_indirect_draws`].
    pub(crate) fn execute_bundle(
        &mut self,
        bundle: &Arc<RenderBundle>,
        validated_draws: Vec<(usize, wgt::BufferAddress)>,
    ) {
        self.commands.push(Command::ExecuteBundle {
            bundle: bundle.clone(),
            validated_draws,
        });
    }

    pub(crate) fn dispatch_workgroups(&mut self, count: [u32; 3]) {
        self.commands.push(Command::DispatchWorkgroups(count));
    }

    pub(crate) fn dispatch_workgroups_indirect(
        &mut self,
        buffer: RecordedBuffer,
        offset: wgt::BufferAddress,
    ) {
        self.commands
            .push(Command::DispatchWorkgroupsIndirect { buffer, offset });
    }

    pub(crate) fn set_dispatch_validation(
        &mut self,
        buffer: &Arc<Buffer>,
        offset: wgt::BufferAddress,
    ) {
        self.commands.push(Command::SetDispatchValidation {
            buffer: buffer.clone(),
            offset,
        });
    }

    pub(crate) fn transition_buffers(
        &mut self,
        barriers: impl IntoIterator<Item = (RecordedBuffer, hal::StateTransition<wgt::BufferUses>)>,
    ) {
        let barriers = barriers.into_iter().collect::<Vec<_>>();
        if !barriers.is_empty() {
            self.commands.push(Command::TransitionBuffers(barriers));
        }
    }

    pub(crate) fn transition_textures(
        &mut self,
        barriers: impl IntoIterator<Item = (Arc<Texture>, PendingTransition<wgt::TextureUses>)>,
    ) {
        let barriers = barriers.into_iter().collect::<Vec<_>>();
        if !barriers.is_empty() {
            self.commands.push(Command::TransitionTextures(barriers));
        }
    }

    pub(crate) fn begin_query(&mut self, query_set: &Arc<QuerySet>, index: u32) {
        self.commands
            .push(Command::BeginQuery(query_set.clone(), index));
    }

    pub(crate) fn end_query(&mut self, query_set: &Arc<QuerySet>, index: u32) {
        self.commands
            .push(Command::EndQuery(query_set.clone(), index));
    }

    pub(crate) fn write_timestamp(&mut self, query_set: &Arc<QuerySet>, index: u32) {
        self.commands
            .push(Command::WriteTimestamp(query_set.clone(), index));
    }

    pub(crate) fn reset_queries(&mut self, query_set: &Arc<QuerySet>, range: Range<u32>) {
        self.commands
            .push(Command::ResetQueries(query_set.clone(), range));
    }

    pub(crate) fn begin_debug_marker(&mut self, label: &str) {
        let label = self.push_label(label);
        self.commands.push(Command::BeginDebugMarker(label));
    }

    pub(crate) fn end_debug_marker(&mut self) {
        self.commands.push(Command::EndDebugMarker);
    }

    pub(crate) fn insert_debug_marker(&mut self, label: &str) {
        let label = self.push_label(label);
        self.commands.push(Command::InsertDebugMarker(label));
    }

    /// `snatch_guard` must have been held since the commands were recorded.
    pub(crate) fn encode<E>(
        self,
        raw: &mut dyn hal::DynCommandEncoder,
        device: &Device,
        indirect_draw_validation_resources: &DrawResources,
        snatch_guard: &SnatchGuard,
    ) -> Result<(), E>
    where
        E: From<DeviceError> + From<InvalidOrDestroyedResourceError>,
    {
        let Self {
            commands,
            data,
            labels,
        } = self;
        let validation = Validation {
            device,
            draw_resources: indirect_draw_validation_resources,
        };

        for command in commands {
            match command {
                Command::BeginRenderPass(desc) => {
                    let color_attachments = desc
                        .color_attachments
                        .iter()
                        .map(|at| {
                            at.as_ref()
                                .map(|at| {
                                    Ok(hal::ColorAttachment {
                                        target: raw_attachment(&at.target, snatch_guard)?,
                                        depth_slice: at.depth_slice,
                                        resolve_target: at
                                            .resolve_target
                                            .as_ref()
                                            .map(|a| raw_attachment(a, snatch_guard))
                                            .transpose()?,
                                        ops: at.ops,
                                        clear_value: at.clear_value,
                                    })
                                })
                                .transpose()
                        })
                        .collect::<Result<Vec<_>, InvalidOrDestroyedResourceError>>()?;
                    let depth_stencil_attachment = desc
                        .depth_stencil_attachment
                        .as_ref()
                        .map(|at| {
                            Ok::<_, InvalidOrDestroyedResourceError>(hal::DepthStencilAttachment {
                                target: raw_attachment(&at.target, snatch_guard)?,
                                depth_ops: at.depth_ops,
                                stencil_ops: at.stencil_ops,
                                depth_read_only: at.depth_read_only,
                                stencil_read_only: at.stencil_read_only,
                                clear_value: at.clear_value,
                            })
                        })
                        .transpose()?;
                    let timestamp_writes = desc
                        .timestamp_writes
                        .as_ref()
                        .map(|tw| raw_timestamp_writes(tw, snatch_guard))
                        .transpose()?;
                    let occlusion_query_set = desc
                        .occlusion_query_set
                        .as_ref()
                        .map(|set| set.try_raw(snatch_guard))
                        .transpose()
                        .map_err(InvalidOrDestroyedResourceError::from)?;
                    let hal_desc = hal::RenderPassDescriptor {
                        label: desc.label.as_deref(),
                        extent: desc.extent,
                        sample_count: desc.sample_count,
                        color_attachments: &color_attachments,
                        depth_stencil_attachment,
                        multiview_mask: desc.multiview_mask,
                        timestamp_writes,
                        occlusion_query_set,
                    };
                    unsafe { raw.begin_render_pass(&hal_desc) }
                        .map_err(|e| device.handle_hal_error(e))?;
                }
                Command::EndRenderPass => unsafe { raw.end_render_pass() },
                Command::BeginComputePass {
                    label,
                    timestamp_writes,
                } => {
                    let timestamp_writes = timestamp_writes
                        .as_ref()
                        .map(|tw| raw_timestamp_writes(tw, snatch_guard))
                        .transpose()?;
                    unsafe {
                        raw.begin_compute_pass(&hal::ComputePassDescriptor {
                            label: label.as_deref(),
                            timestamp_writes,
                        })
                    }
                }
                Command::EndComputePass => unsafe { raw.end_compute_pass() },
                Command::SetRenderPipeline(pipeline) => {
                    let pipeline = pipeline
                        .raw()
                        .map_err(InvalidOrDestroyedResourceError::from)?;
                    unsafe { raw.set_render_pipeline(pipeline) }
                }
                Command::SetComputePipeline(pipeline) => {
                    let pipeline = pipeline
                        .raw()
                        .map_err(InvalidOrDestroyedResourceError::from)?;
                    unsafe { raw.set_compute_pipeline(pipeline) }
                }
                Command::SetBindGroup {
                    layout,
                    index,
                    bind_group,
                    dynamic_offsets,
                } => {
                    let layout = layout
                        .raw()
                        .map_err(InvalidOrDestroyedResourceError::from)?;
                    let bind_group = bind_group.try_raw(snatch_guard)?;
                    unsafe { raw.set_bind_group(layout, index, bind_group, &data[dynamic_offsets]) }
                }
                Command::SetImmediates {
                    layout,
                    offset_bytes,
                    data: range,
                } => {
                    let layout = layout
                        .raw()
                        .map_err(InvalidOrDestroyedResourceError::from)?;
                    unsafe { raw.set_immediates(layout, offset_bytes, &data[range]) }
                }
                Command::SetIndexBuffer {
                    buffer,
                    offset,
                    size,
                    format,
                } => {
                    let buffer = raw_user_buffer(&buffer, snatch_guard)?;
                    // SAFETY: The binding was validated before it was recorded.
                    let binding = hal::BufferBinding::new_unchecked(buffer, offset, size);
                    unsafe { raw.set_index_buffer(binding, format) }
                }
                Command::SetVertexBuffer {
                    slot,
                    buffer,
                    offset,
                    size,
                } => {
                    let buffer = raw_user_buffer(&buffer, snatch_guard)?;
                    // SAFETY: The binding was validated before it was recorded.
                    let binding = hal::BufferBinding::new_unchecked(buffer, offset, size);
                    unsafe { raw.set_vertex_buffer(slot, binding) }
                }
                Command::SetViewport { rect, depth_range } => unsafe {
                    raw.set_viewport(&rect, depth_range)
                },
                Command::SetScissorRect(rect) => unsafe { raw.set_scissor_rect(&rect) },
                Command::SetStencilReference(value) => unsafe { raw.set_stencil_reference(value) },
                Command::SetBlendConstants(color) => unsafe { raw.set_blend_constants(&color) },
                Command::Draw {
                    first_vertex,
                    vertex_count,
                    first_instance,
                    instance_count,
                } => unsafe {
                    raw.draw(first_vertex, vertex_count, first_instance, instance_count)
                },
                Command::DrawIndexed {
                    first_index,
                    index_count,
                    base_vertex,
                    first_instance,
                    instance_count,
                } => unsafe {
                    raw.draw_indexed(
                        first_index,
                        index_count,
                        base_vertex,
                        first_instance,
                        instance_count,
                    )
                },
                Command::DrawMeshTasks([x, y, z]) => unsafe { raw.draw_mesh_tasks(x, y, z) },
                Command::DrawIndirect {
                    kind,
                    buffer,
                    offset,
                    draw_count,
                } => {
                    let buffer = validation.raw_buffer(&buffer, snatch_guard)?;
                    match kind {
                        IndirectDrawKind::Draw => unsafe {
                            raw.draw_indirect(buffer, offset, draw_count)
                        },
                        IndirectDrawKind::DrawIndexed => unsafe {
                            raw.draw_indexed_indirect(buffer, offset, draw_count)
                        },
                        IndirectDrawKind::DrawMeshTasks => unsafe {
                            raw.draw_mesh_tasks_indirect(buffer, offset, draw_count)
                        },
                    }
                }
                Command::DrawIndirectCount {
                    kind,
                    buffer,
                    offset,
                    count_buffer,
                    count_offset,
                    max_count,
                } => {
                    let buffer = raw_user_buffer(&buffer, snatch_guard)?;
                    let count_buffer = raw_user_buffer(&count_buffer, snatch_guard)?;
                    match kind {
                        IndirectDrawKind::Draw => unsafe {
                            raw.draw_indirect_count(
                                buffer,
                                offset,
                                count_buffer,
                                count_offset,
                                max_count,
                            )
                        },
                        IndirectDrawKind::DrawIndexed => unsafe {
                            raw.draw_indexed_indirect_count(
                                buffer,
                                offset,
                                count_buffer,
                                count_offset,
                                max_count,
                            )
                        },
                        IndirectDrawKind::DrawMeshTasks => unsafe {
                            raw.draw_mesh_tasks_indirect_count(
                                buffer,
                                offset,
                                count_buffer,
                                count_offset,
                                max_count,
                            )
                        },
                    }
                }
                Command::ExecuteBundle {
                    bundle,
                    validated_draws,
                } => unsafe {
                    bundle.execute(
                        raw,
                        indirect_draw_validation_resources,
                        &validated_draws,
                        snatch_guard,
                    )?
                },
                Command::DispatchWorkgroups(count) => unsafe { raw.dispatch_workgroups(count) },
                Command::DispatchWorkgroupsIndirect { buffer, offset } => {
                    let buffer = validation.raw_buffer(&buffer, snatch_guard)?;
                    unsafe { raw.dispatch_workgroups_indirect(buffer, offset) }
                }
                Command::SetDispatchValidation { buffer, offset } => {
                    let params = validation
                        .dispatch()
                        .params(&device.limits, offset, buffer.size);
                    let src_bind_group = buffer
                        .indirect_validation_bind_groups
                        .get(snatch_guard)
                        .ok_or_else(|| DestroyedResourceError(buffer.error_ident()))
                        .map_err(InvalidOrDestroyedResourceError::from)?;
                    unsafe {
                        raw.set_compute_pipeline(params.pipeline);
                        raw.set_immediates(
                            params.pipeline_layout,
                            0,
                            &[params.offset_remainder as u32 / 4],
                        );
                        raw.set_bind_group(params.pipeline_layout, 0, params.dst_bind_group, &[]);
                        raw.set_bind_group(
                            params.pipeline_layout,
                            1,
                            src_bind_group.dispatch.as_ref(),
                            &[params.aligned_offset as u32],
                        );
                    }
                }
                Command::TransitionBuffers(barriers) => {
                    let barriers = barriers
                        .iter()
                        .map(|(buffer, usage)| {
                            Ok(hal::BufferBarrier {
                                buffer: validation.raw_buffer(buffer, snatch_guard)?,
                                usage: usage.clone(),
                            })
                        })
                        .collect::<Result<Vec<_>, InvalidOrDestroyedResourceError>>()?;
                    unsafe { raw.transition_buffers(&barriers) };
                }
                Command::TransitionTextures(barriers) => {
                    let (textures, transitions): (Vec<_>, Vec<_>) = barriers.into_iter().unzip();
                    let textures = textures
                        .iter()
                        .map(|texture| texture.try_inner(snatch_guard))
                        .collect::<Result<Vec<_>, _>>()?;
                    let barriers = transitions
                        .into_iter()
                        .zip(textures)
                        .map(|(transition, texture)| transition.into_hal(texture.raw()))
                        .collect::<Vec<_>>();
                    unsafe { raw.transition_textures(&barriers) };
                }
                Command::BeginQuery(set, index) => {
                    let set = raw_query_set(&set, snatch_guard)?;
                    unsafe { raw.begin_query(set, index) }
                }
                Command::EndQuery(set, index) => {
                    let set = raw_query_set(&set, snatch_guard)?;
                    unsafe { raw.end_query(set, index) }
                }
                Command::WriteTimestamp(set, index) => {
                    let set = raw_query_set(&set, snatch_guard)?;
                    unsafe { raw.write_timestamp(set, index) }
                }
                Command::ResetQueries(set, range) => {
                    let set = raw_query_set(&set, snatch_guard)?;
                    unsafe { raw.reset_queries(set, range) }
                }
                Command::BeginDebugMarker(range) => unsafe {
                    raw.begin_debug_marker(&labels[range])
                },
                Command::EndDebugMarker => unsafe { raw.end_debug_marker() },
                Command::InsertDebugMarker(range) => unsafe {
                    raw.insert_debug_marker(&labels[range])
                },
            }
        }

        Ok(())
    }
}

struct Validation<'a> {
    device: &'a Device,
    draw_resources: &'a DrawResources,
}

impl<'a> Validation<'a> {
    fn dispatch(&self) -> &'a crate::indirect_validation::Dispatch {
        &self
            .device
            .indirect_validation
            .as_ref()
            .expect("indirect validation was recorded")
            .dispatch
    }

    fn raw_buffer<'b>(
        &self,
        buffer: &'b RecordedBuffer,
        snatch_guard: &'b SnatchGuard,
    ) -> Result<&'b dyn hal::DynBuffer, InvalidOrDestroyedResourceError>
    where
        'a: 'b,
    {
        Ok(match *buffer {
            RecordedBuffer::Buffer(ref buffer) => raw_user_buffer(buffer, snatch_guard)?,
            RecordedBuffer::DrawValidation(index) => self.draw_resources.get_dst_buffer(index),
            RecordedBuffer::DispatchValidation => self.dispatch().dst_buffer(),
        })
    }
}

fn raw_user_buffer<'a>(
    buffer: &'a Arc<Buffer>,
    snatch_guard: &'a SnatchGuard,
) -> Result<&'a dyn hal::DynBuffer, InvalidOrDestroyedResourceError> {
    Ok(buffer.try_raw(snatch_guard)?)
}

fn raw_query_set<'a>(
    query_set: &'a Arc<QuerySet>,
    snatch_guard: &'a SnatchGuard,
) -> Result<&'a dyn hal::DynQuerySet, InvalidOrDestroyedResourceError> {
    Ok(query_set.try_raw(snatch_guard)?)
}

fn raw_attachment<'a>(
    attachment: &'a RecordedAttachment,
    snatch_guard: &'a SnatchGuard,
) -> Result<hal::Attachment<'a, dyn hal::DynTextureView>, InvalidOrDestroyedResourceError> {
    Ok(hal::Attachment {
        view: attachment.view.try_raw(snatch_guard)?,
        usage: attachment.usage,
    })
}

fn raw_timestamp_writes<'a>(
    timestamp_writes: &'a RecordedTimestampWrites,
    snatch_guard: &'a SnatchGuard,
) -> Result<hal::PassTimestampWrites<'a, dyn hal::DynQuerySet>, InvalidOrDestroyedResourceError> {
    Ok(hal::PassTimestampWrites {
        query_set: raw_query_set(&timestamp_writes.query_set, snatch_guard)?,
        beginning_of_pass_write_index: timestamp_writes.beginning_of_pass_write_index,
        end_of_pass_write_index: timestamp_writes.end_of_pass_write_index,
    })
}
