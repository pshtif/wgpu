//! Kosuno experiment (`single`, see `kosuno_patch`): a command encoder that
//! records a pass instead of encoding it, so wgpu-core can encode the pass's
//! barriers first and then replay the pass into the same command buffer.
//!
//! Resources are held as `&'static` references made from shorter borrows.
//! That is sound only because wgpu-core replays before the borrows end:
//! every resource a pass touches is kept alive by the pass's trackers and
//! the snatch guard held across `encode_commands`.

use alloc::{boxed::Box, string::String, vec::Vec};
use core::ops::Range;

use crate::{
    AccelerationStructureBarrier, BufferBarrier, BufferBinding, BufferCopy, BufferTextureCopy,
    BuildAccelerationStructureDescriptor, ColorAttachment, ComputePassDescriptor,
    DepthStencilAttachment, DeviceError, DynRayTracingPipeline, Label, MemoryRange,
    PassTimestampWrites, RayTracingPassDescriptor, Rect, RenderPassDescriptor, StateTransition,
    TextureBarrier, TextureCopy,
};

use super::{
    DynAccelerationStructure, DynBindGroup, DynBuffer, DynCommandBuffer, DynCommandEncoder,
    DynComputePipeline, DynPipelineLayout, DynQuerySet, DynRenderPipeline, DynTexture,
    DynTextureView,
};

type B = &'static dyn DynBuffer;
type T = &'static dyn DynTexture;
type Q = &'static dyn DynQuerySet;

/// Extends a borrow to `'static`. See the module docs for why it holds.
macro_rules! st {
    ($e:expr) => {
        unsafe { core::mem::transmute::<_, &'static _>($e) }
    };
}

fn attachment(a: &crate::Attachment<'_, dyn DynTextureView>) -> crate::Attachment<'static, dyn DynTextureView> {
    crate::Attachment { view: st!(a.view), usage: a.usage }
}
fn color(a: &ColorAttachment<'_, dyn DynTextureView>) -> ColorAttachment<'static, dyn DynTextureView> {
    ColorAttachment {
        target: attachment(&a.target),
        depth_slice: a.depth_slice,
        resolve_target: a.resolve_target.as_ref().map(attachment),
        ops: a.ops,
        clear_value: a.clear_value,
    }
}
fn depth(a: &DepthStencilAttachment<'_, dyn DynTextureView>) -> DepthStencilAttachment<'static, dyn DynTextureView> {
    DepthStencilAttachment { target: attachment(&a.target), depth_ops: a.depth_ops, stencil_ops: a.stencil_ops, clear_value: a.clear_value }
}
fn timestamps(t: &PassTimestampWrites<'_, dyn DynQuerySet>) -> PassTimestampWrites<'static, dyn DynQuerySet> {
    PassTimestampWrites {
        query_set: st!(t.query_set),
        beginning_of_pass_write_index: t.beginning_of_pass_write_index,
        end_of_pass_write_index: t.end_of_pass_write_index,
    }
}

enum Op {
    TransitionBuffers(Vec<(B, StateTransition<wgt::BufferUses>)>),
    TransitionTextures(Vec<(T, wgt::ImageSubresourceRange, StateTransition<wgt::TextureUses>)>),
    ClearBuffer(B, MemoryRange),
    CopyBufferToBuffer(B, B, Vec<BufferCopy>),
    CopyTextureToTexture(T, wgt::TextureUses, T, Vec<TextureCopy>),
    CopyBufferToTexture(B, T, Vec<BufferTextureCopy>),
    CopyTextureToBuffer(T, wgt::TextureUses, B, Vec<BufferTextureCopy>),
    SetBindGroup(&'static dyn DynPipelineLayout, u32, &'static dyn DynBindGroup, Vec<wgt::DynamicOffset>),
    SetImmediates(&'static dyn DynPipelineLayout, u32, Vec<u32>),
    InsertDebugMarker(String),
    BeginDebugMarker(String),
    EndDebugMarker,
    BeginQuery(Q, u32),
    EndQuery(Q, u32),
    WriteTimestamp(Q, u32),
    ResetQueries(Q, Range<u32>),
    CopyQueryResults(Q, Range<u32>, B, wgt::BufferAddress, wgt::BufferSize),
    BeginRenderPass(Box<RenderPass>),
    EndRenderPass,
    SetRenderPipeline(&'static dyn DynRenderPipeline),
    SetIndexBuffer(B, wgt::BufferAddress, Option<wgt::BufferSize>, wgt::IndexFormat),
    SetVertexBuffer(u32, B, wgt::BufferAddress, Option<wgt::BufferSize>),
    SetViewport(Rect<f32>, Range<f32>),
    SetScissorRect(Rect<u32>),
    SetStencilReference(u32),
    SetBlendConstants([f32; 4]),
    Draw(u32, u32, u32, u32),
    DrawIndexed(u32, u32, i32, u32, u32),
    DrawMeshTasks(u32, u32, u32),
    DrawIndirect(B, wgt::BufferAddress, u32),
    DrawIndexedIndirect(B, wgt::BufferAddress, u32),
    DrawMeshTasksIndirect(B, wgt::BufferAddress, u32),
    DrawIndirectCount(B, wgt::BufferAddress, B, wgt::BufferAddress, u32),
    DrawIndexedIndirectCount(B, wgt::BufferAddress, B, wgt::BufferAddress, u32),
    DrawMeshTasksIndirectCount(B, wgt::BufferAddress, B, wgt::BufferAddress, u32),
    BeginComputePass(Option<String>, Option<PassTimestampWrites<'static, dyn DynQuerySet>>),
    EndComputePass,
    SetComputePipeline(&'static dyn DynComputePipeline),
    DispatchWorkgroups([u32; 3]),
    DispatchWorkgroupsIndirect(B, wgt::BufferAddress),
}

struct RenderPass {
    label: Option<String>,
    extent: wgt::Extent3d,
    sample_count: u32,
    color_attachments: Vec<Option<ColorAttachment<'static, dyn DynTextureView>>>,
    depth_stencil_attachment: Option<DepthStencilAttachment<'static, dyn DynTextureView>>,
    multiview_mask: Option<core::num::NonZeroU32>,
    timestamp_writes: Option<PassTimestampWrites<'static, dyn DynQuerySet>>,
    occlusion_query_set: Option<Q>,
}

/// Records a pass; [`PassRecorder::replay`] encodes it into a real encoder.
#[derive(Default)]
pub struct PassRecorder {
    ops: Vec<Op>,
}

impl core::fmt::Debug for PassRecorder {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PassRecorder").field("ops", &self.ops.len()).finish()
    }
}

crate::impl_dyn_resource!(PassRecorder);

impl PassRecorder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Encodes the recorded commands into `target`, an open encoder.
    ///
    /// # Safety
    ///
    /// Every resource recorded must still be alive (see the module docs).
    pub unsafe fn replay(self, target: &mut dyn DynCommandEncoder) {
        for op in self.ops {
            unsafe {
                match op {
                    Op::TransitionBuffers(v) => {
                        let barriers: Vec<_> =
                            v.into_iter().map(|(buffer, usage)| BufferBarrier { buffer, usage }).collect();
                        target.transition_buffers(&barriers);
                    }
                    Op::TransitionTextures(v) => {
                        let barriers: Vec<_> = v
                            .into_iter()
                            .map(|(texture, range, usage)| TextureBarrier { texture, range, usage })
                            .collect();
                        target.transition_textures(&barriers);
                    }
                    Op::ClearBuffer(b, r) => target.clear_buffer(b, r),
                    Op::CopyBufferToBuffer(s, d, r) => target.copy_buffer_to_buffer(s, d, &r),
                    Op::CopyTextureToTexture(s, u, d, r) => target.copy_texture_to_texture(s, u, d, &r),
                    Op::CopyBufferToTexture(s, d, r) => target.copy_buffer_to_texture(s, d, &r),
                    Op::CopyTextureToBuffer(s, u, d, r) => target.copy_texture_to_buffer(s, u, d, &r),
                    Op::SetBindGroup(l, i, g, o) => target.set_bind_group(l, i, g, &o),
                    Op::SetImmediates(l, o, d) => target.set_immediates(l, o, &d),
                    Op::InsertDebugMarker(s) => target.insert_debug_marker(&s),
                    Op::BeginDebugMarker(s) => target.begin_debug_marker(&s),
                    Op::EndDebugMarker => target.end_debug_marker(),
                    Op::BeginQuery(q, i) => target.begin_query(q, i),
                    Op::EndQuery(q, i) => target.end_query(q, i),
                    Op::WriteTimestamp(q, i) => target.write_timestamp(q, i),
                    Op::ResetQueries(q, r) => target.reset_queries(q, r),
                    Op::CopyQueryResults(q, r, b, o, s) => target.copy_query_results(q, r, b, o, s),
                    Op::BeginRenderPass(p) => {
                        let desc = RenderPassDescriptor {
                            label: p.label.as_deref(),
                            extent: p.extent,
                            sample_count: p.sample_count,
                            color_attachments: &p.color_attachments,
                            depth_stencil_attachment: p.depth_stencil_attachment.as_ref().map(depth),
                            multiview_mask: p.multiview_mask,
                            timestamp_writes: p.timestamp_writes.as_ref().map(timestamps),
                            occlusion_query_set: p.occlusion_query_set,
                        };
                        // An error here is a device loss; the real encoder
                        // reports it again at end_encoding / submit.
                        let _ = target.begin_render_pass(&desc);
                    }
                    Op::EndRenderPass => target.end_render_pass(),
                    Op::SetRenderPipeline(p) => target.set_render_pipeline(p),
                    Op::SetIndexBuffer(buffer, offset, size, f) => {
                        target.set_index_buffer(BufferBinding { buffer, offset, size }, f)
                    }
                    Op::SetVertexBuffer(i, buffer, offset, size) => {
                        target.set_vertex_buffer(i, BufferBinding { buffer, offset, size })
                    }
                    Op::SetViewport(r, d) => target.set_viewport(&r, d),
                    Op::SetScissorRect(r) => target.set_scissor_rect(&r),
                    Op::SetStencilReference(v) => target.set_stencil_reference(v),
                    Op::SetBlendConstants(c) => target.set_blend_constants(&c),
                    Op::Draw(a, b, c, d) => target.draw(a, b, c, d),
                    Op::DrawIndexed(a, b, c, d, e) => target.draw_indexed(a, b, c, d, e),
                    Op::DrawMeshTasks(x, y, z) => target.draw_mesh_tasks(x, y, z),
                    Op::DrawIndirect(b, o, n) => target.draw_indirect(b, o, n),
                    Op::DrawIndexedIndirect(b, o, n) => target.draw_indexed_indirect(b, o, n),
                    Op::DrawMeshTasksIndirect(b, o, n) => target.draw_mesh_tasks_indirect(b, o, n),
                    Op::DrawIndirectCount(b, o, c, co, m) => target.draw_indirect_count(b, o, c, co, m),
                    Op::DrawIndexedIndirectCount(b, o, c, co, m) => {
                        target.draw_indexed_indirect_count(b, o, c, co, m)
                    }
                    Op::DrawMeshTasksIndirectCount(b, o, c, co, m) => {
                        target.draw_mesh_tasks_indirect_count(b, o, c, co, m)
                    }
                    Op::BeginComputePass(label, timestamp_writes) => {
                        target.begin_compute_pass(&ComputePassDescriptor {
                            label: label.as_deref(),
                            timestamp_writes,
                        })
                    }
                    Op::EndComputePass => target.end_compute_pass(),
                    Op::SetComputePipeline(p) => target.set_compute_pipeline(p),
                    Op::DispatchWorkgroups(c) => target.dispatch_workgroups(c),
                    Op::DispatchWorkgroupsIndirect(b, o) => target.dispatch_workgroups_indirect(b, o),
                }
            }
        }
    }
}

const NOT_IN_PASS: &str = "not recorded inside a pass";

impl DynCommandEncoder for PassRecorder {
    unsafe fn begin_encoding(&mut self, _label: Label) -> Result<(), DeviceError> {
        unreachable!("{NOT_IN_PASS}")
    }
    unsafe fn discard_encoding(&mut self) {
        self.ops.clear();
    }
    unsafe fn end_encoding(&mut self) -> Result<Box<dyn DynCommandBuffer>, DeviceError> {
        unreachable!("{NOT_IN_PASS}")
    }
    unsafe fn reset_all(&mut self, _command_buffers: Vec<Box<dyn DynCommandBuffer>>) {
        unreachable!("{NOT_IN_PASS}")
    }

    unsafe fn transition_buffers(&mut self, barriers: &[BufferBarrier<'_, dyn DynBuffer>]) {
        let v = barriers.iter().map(|b| (st!(b.buffer), b.usage.clone())).collect();
        self.ops.push(Op::TransitionBuffers(v));
    }
    unsafe fn transition_textures(&mut self, barriers: &[TextureBarrier<'_, dyn DynTexture>]) {
        let v = barriers.iter().map(|b| (st!(b.texture), b.range, b.usage.clone())).collect();
        self.ops.push(Op::TransitionTextures(v));
    }
    unsafe fn clear_buffer(&mut self, buffer: &dyn DynBuffer, range: MemoryRange) {
        self.ops.push(Op::ClearBuffer(st!(buffer), range));
    }
    unsafe fn copy_buffer_to_buffer(&mut self, src: &dyn DynBuffer, dst: &dyn DynBuffer, regions: &[BufferCopy]) {
        self.ops.push(Op::CopyBufferToBuffer(st!(src), st!(dst), regions.to_vec()));
    }
    unsafe fn copy_texture_to_texture(
        &mut self,
        src: &dyn DynTexture,
        src_usage: wgt::TextureUses,
        dst: &dyn DynTexture,
        regions: &[TextureCopy],
    ) {
        self.ops.push(Op::CopyTextureToTexture(st!(src), src_usage, st!(dst), regions.to_vec()));
    }
    unsafe fn copy_buffer_to_texture(&mut self, src: &dyn DynBuffer, dst: &dyn DynTexture, regions: &[BufferTextureCopy]) {
        self.ops.push(Op::CopyBufferToTexture(st!(src), st!(dst), regions.to_vec()));
    }
    unsafe fn copy_texture_to_buffer(
        &mut self,
        src: &dyn DynTexture,
        src_usage: wgt::TextureUses,
        dst: &dyn DynBuffer,
        regions: &[BufferTextureCopy],
    ) {
        self.ops.push(Op::CopyTextureToBuffer(st!(src), src_usage, st!(dst), regions.to_vec()));
    }
    unsafe fn set_bind_group(
        &mut self,
        layout: &dyn DynPipelineLayout,
        index: u32,
        group: &dyn DynBindGroup,
        dynamic_offsets: &[wgt::DynamicOffset],
    ) {
        self.ops.push(Op::SetBindGroup(st!(layout), index, st!(group), dynamic_offsets.to_vec()));
    }
    unsafe fn set_immediates(&mut self, layout: &dyn DynPipelineLayout, offset_bytes: u32, data: &[u32]) {
        self.ops.push(Op::SetImmediates(st!(layout), offset_bytes, data.to_vec()));
    }
    unsafe fn insert_debug_marker(&mut self, label: &str) {
        self.ops.push(Op::InsertDebugMarker(label.into()));
    }
    unsafe fn begin_debug_marker(&mut self, group_label: &str) {
        self.ops.push(Op::BeginDebugMarker(group_label.into()));
    }
    unsafe fn end_debug_marker(&mut self) {
        self.ops.push(Op::EndDebugMarker);
    }
    unsafe fn begin_query(&mut self, set: &dyn DynQuerySet, index: u32) {
        self.ops.push(Op::BeginQuery(st!(set), index));
    }
    unsafe fn end_query(&mut self, set: &dyn DynQuerySet, index: u32) {
        self.ops.push(Op::EndQuery(st!(set), index));
    }
    unsafe fn write_timestamp(&mut self, set: &dyn DynQuerySet, index: u32) {
        self.ops.push(Op::WriteTimestamp(st!(set), index));
    }
    unsafe fn reset_queries(&mut self, set: &dyn DynQuerySet, range: Range<u32>) {
        self.ops.push(Op::ResetQueries(st!(set), range));
    }
    unsafe fn copy_query_results(
        &mut self,
        set: &dyn DynQuerySet,
        range: Range<u32>,
        buffer: &dyn DynBuffer,
        offset: wgt::BufferAddress,
        stride: wgt::BufferSize,
    ) {
        self.ops.push(Op::CopyQueryResults(st!(set), range, st!(buffer), offset, stride));
    }

    unsafe fn begin_render_pass(
        &mut self,
        desc: &RenderPassDescriptor<dyn DynQuerySet, dyn DynTextureView>,
    ) -> Result<(), DeviceError> {
        let pass = RenderPass {
            label: desc.label.map(String::from),
            extent: desc.extent,
            sample_count: desc.sample_count,
            color_attachments: desc
                .color_attachments
                .iter()
                .map(|a| a.as_ref().map(color))
                .collect(),
            depth_stencil_attachment: desc.depth_stencil_attachment.as_ref().map(depth),
            multiview_mask: desc.multiview_mask,
            timestamp_writes: desc.timestamp_writes.as_ref().map(timestamps),
            occlusion_query_set: desc.occlusion_query_set.map(|q| st!(q)),
        };
        self.ops.push(Op::BeginRenderPass(Box::new(pass)));
        Ok(())
    }
    unsafe fn end_render_pass(&mut self) {
        self.ops.push(Op::EndRenderPass);
    }
    unsafe fn set_render_pipeline(&mut self, pipeline: &dyn DynRenderPipeline) {
        self.ops.push(Op::SetRenderPipeline(st!(pipeline)));
    }
    unsafe fn set_index_buffer<'a>(&mut self, binding: BufferBinding<'a, dyn DynBuffer>, format: wgt::IndexFormat) {
        self.ops.push(Op::SetIndexBuffer(st!(binding.buffer), binding.offset, binding.size, format));
    }
    unsafe fn set_vertex_buffer<'a>(&mut self, index: u32, binding: BufferBinding<'a, dyn DynBuffer>) {
        self.ops.push(Op::SetVertexBuffer(index, st!(binding.buffer), binding.offset, binding.size));
    }
    unsafe fn set_viewport(&mut self, rect: &Rect<f32>, depth_range: Range<f32>) {
        self.ops.push(Op::SetViewport(rect.clone(), depth_range));
    }
    unsafe fn set_scissor_rect(&mut self, rect: &Rect<u32>) {
        self.ops.push(Op::SetScissorRect(rect.clone()));
    }
    unsafe fn set_stencil_reference(&mut self, value: u32) {
        self.ops.push(Op::SetStencilReference(value));
    }
    unsafe fn set_blend_constants(&mut self, color: &[f32; 4]) {
        self.ops.push(Op::SetBlendConstants(*color));
    }
    unsafe fn draw(&mut self, first_vertex: u32, vertex_count: u32, first_instance: u32, instance_count: u32) {
        self.ops.push(Op::Draw(first_vertex, vertex_count, first_instance, instance_count));
    }
    unsafe fn draw_indexed(
        &mut self,
        first_index: u32,
        index_count: u32,
        base_vertex: i32,
        first_instance: u32,
        instance_count: u32,
    ) {
        self.ops.push(Op::DrawIndexed(first_index, index_count, base_vertex, first_instance, instance_count));
    }
    unsafe fn draw_mesh_tasks(&mut self, x: u32, y: u32, z: u32) {
        self.ops.push(Op::DrawMeshTasks(x, y, z));
    }
    unsafe fn draw_indirect(&mut self, buffer: &dyn DynBuffer, offset: wgt::BufferAddress, draw_count: u32) {
        self.ops.push(Op::DrawIndirect(st!(buffer), offset, draw_count));
    }
    unsafe fn draw_indexed_indirect(&mut self, buffer: &dyn DynBuffer, offset: wgt::BufferAddress, draw_count: u32) {
        self.ops.push(Op::DrawIndexedIndirect(st!(buffer), offset, draw_count));
    }
    unsafe fn draw_mesh_tasks_indirect(&mut self, buffer: &dyn DynBuffer, offset: wgt::BufferAddress, draw_count: u32) {
        self.ops.push(Op::DrawMeshTasksIndirect(st!(buffer), offset, draw_count));
    }
    unsafe fn draw_indirect_count(
        &mut self,
        buffer: &dyn DynBuffer,
        offset: wgt::BufferAddress,
        count_buffer: &dyn DynBuffer,
        count_offset: wgt::BufferAddress,
        max_count: u32,
    ) {
        self.ops.push(Op::DrawIndirectCount(st!(buffer), offset, st!(count_buffer), count_offset, max_count));
    }
    unsafe fn draw_indexed_indirect_count(
        &mut self,
        buffer: &dyn DynBuffer,
        offset: wgt::BufferAddress,
        count_buffer: &dyn DynBuffer,
        count_offset: wgt::BufferAddress,
        max_count: u32,
    ) {
        self.ops.push(Op::DrawIndexedIndirectCount(st!(buffer), offset, st!(count_buffer), count_offset, max_count));
    }
    unsafe fn draw_mesh_tasks_indirect_count(
        &mut self,
        buffer: &dyn DynBuffer,
        offset: wgt::BufferAddress,
        count_buffer: &dyn DynBuffer,
        count_offset: wgt::BufferAddress,
        max_count: u32,
    ) {
        self.ops.push(Op::DrawMeshTasksIndirectCount(st!(buffer), offset, st!(count_buffer), count_offset, max_count));
    }

    unsafe fn begin_compute_pass(&mut self, desc: &ComputePassDescriptor<dyn DynQuerySet>) {
        let timestamp_writes = desc.timestamp_writes.as_ref().map(timestamps);
        self.ops.push(Op::BeginComputePass(desc.label.map(String::from), timestamp_writes));
    }
    unsafe fn end_compute_pass(&mut self) {
        self.ops.push(Op::EndComputePass);
    }
    unsafe fn set_compute_pipeline(&mut self, pipeline: &dyn DynComputePipeline) {
        self.ops.push(Op::SetComputePipeline(st!(pipeline)));
    }
    unsafe fn dispatch_workgroups(&mut self, count: [u32; 3]) {
        self.ops.push(Op::DispatchWorkgroups(count));
    }
    unsafe fn dispatch_workgroups_indirect(&mut self, buffer: &dyn DynBuffer, offset: wgt::BufferAddress) {
        self.ops.push(Op::DispatchWorkgroupsIndirect(st!(buffer), offset));
    }

    unsafe fn begin_ray_tracing_pass(&mut self, _desc: &RayTracingPassDescriptor) {
        unreachable!("{NOT_IN_PASS}")
    }
    unsafe fn end_ray_tracing_pass(&mut self) {
        unreachable!("{NOT_IN_PASS}")
    }
    unsafe fn trace_rays(
        &mut self,
        _count: [u32; 3],
        _ray_generation_group_data: crate::PipelineGroupData<dyn DynBuffer>,
        _miss_group_data: crate::PipelineGroupData<dyn DynBuffer>,
        _intersection_group_data: crate::PipelineGroupData<dyn DynBuffer>,
    ) {
        unreachable!("{NOT_IN_PASS}")
    }
    unsafe fn set_ray_tracing_pipeline(&mut self, _pipeline: &dyn DynRayTracingPipeline) {
        unreachable!("{NOT_IN_PASS}")
    }
    unsafe fn build_acceleration_structures<'a>(
        &mut self,
        _descriptors: &'a [BuildAccelerationStructureDescriptor<'a, dyn DynBuffer, dyn DynAccelerationStructure>],
    ) {
        unreachable!("{NOT_IN_PASS}")
    }
    unsafe fn place_acceleration_structure_barrier(&mut self, _barrier: AccelerationStructureBarrier) {
        unreachable!("{NOT_IN_PASS}")
    }
    unsafe fn copy_acceleration_structure_to_acceleration_structure(
        &mut self,
        _src: &dyn DynAccelerationStructure,
        _dst: &dyn DynAccelerationStructure,
        _copy: wgt::AccelerationStructureCopy,
    ) {
        unreachable!("{NOT_IN_PASS}")
    }
    unsafe fn read_acceleration_structure_compact_size(
        &mut self,
        _acceleration_structure: &dyn DynAccelerationStructure,
        _buf: &dyn DynBuffer,
    ) {
        unreachable!("{NOT_IN_PASS}")
    }
    unsafe fn set_acceleration_structure_dependencies(
        &self,
        _command_buffers: &[Box<dyn DynCommandBuffer>],
        _dependencies: &[&dyn DynAccelerationStructure],
    ) {
        unreachable!("{NOT_IN_PASS}")
    }
}
