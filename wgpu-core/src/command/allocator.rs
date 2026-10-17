use alloc::{boxed::Box, vec::Vec};

use crate::lock::{rank, Mutex};

/// A pool of free [`wgpu_hal::CommandEncoder`]s, owned by a `Device`.
///
/// Each encoder in this list is in the "closed" state.
///
/// Since a raw [`CommandEncoder`][ce] is itself a pool for allocating
/// raw [`CommandBuffer`][cb]s, this is a pool of pools.
///
/// [`wgpu_hal::CommandEncoder`]: hal::CommandEncoder
/// [ce]: hal::CommandEncoder
/// [cb]: hal::Api::CommandBuffer
pub(crate) struct CommandAllocator {
    free_encoders: Mutex<Vec<Box<dyn hal::DynCommandEncoder>>>,
    /// Kosuno experiment: when each free encoder was released (same order).
    released_at: Mutex<Vec<std::time::Instant>>,
    /// Kosuno experiment (`adaptive4`): the calm epoch each encoder in use
    /// was acquired in, by its address.
    acquired_in: Mutex<std::collections::HashMap<usize, u32>>,
    /// The epoch each free encoder was acquired in (same order as the list).
    free_epochs: Mutex<Vec<u32>>,
}

impl CommandAllocator {
    pub(crate) fn new() -> Self {
        Self {
            free_encoders: Mutex::new(rank::COMMAND_ALLOCATOR_FREE_ENCODERS, Vec::new()),
            released_at: Mutex::new(rank::COMMAND_ALLOCATOR_FREE_ENCODERS, Vec::new()),
            acquired_in: Mutex::new(rank::COMMAND_ALLOCATOR_FREE_ENCODERS, Default::default()),
            free_epochs: Mutex::new(rank::COMMAND_ALLOCATOR_FREE_ENCODERS, Vec::new()),
        }
    }

    /// Return a fresh [`wgpu_hal::CommandEncoder`] in the "closed" state.
    ///
    /// If we have free encoders in the pool, take one of those. Otherwise,
    /// create a new one on `device`.
    ///
    /// [`wgpu_hal::CommandEncoder`]: hal::CommandEncoder
    pub(crate) fn acquire_encoder(
        &self,
        device: &dyn hal::DynDevice,
        queue: &dyn hal::DynQueue,
    ) -> Result<Box<dyn hal::DynCommandEncoder>, hal::DeviceError> {
        let mut free_encoders = self.free_encoders.lock();
        if hal::kosuno_patch::age_ms().is_some() {
            self.released_at.lock().pop();
        }
        let mut popped = free_encoders.pop();
        if let Some(epoch) = hal::kosuno_patch::destroy_epoch() {
            let was = self.free_epochs.lock().pop();
            // Idle across the calm epoch: replace it with a fresh pool.
            if popped.is_some() && was.is_some_and(|w| w < epoch) {
                drop(popped.take());
            }
        }
        let encoder = match popped {
            Some(encoder) => Ok(encoder),
            None => unsafe {
                let hal_desc = hal::CommandEncoderDescriptor { label: None, queue };
                device.create_command_encoder(&hal_desc)
            },
        };
        if let (Some(epoch), Ok(e)) = (hal::kosuno_patch::destroy_epoch(), &encoder) {
            let key = &**e as *const dyn hal::DynCommandEncoder as *const () as usize;
            self.acquired_in.lock().insert(key, epoch);
        }
        encoder
    }

    /// Add `encoder` back to the free pool.
    pub(crate) fn release_encoder(&self, encoder: Box<dyn hal::DynCommandEncoder>) {
        // Kosuno experiment (`adaptive4`): an encoder acquired before the
        // calm epoch goes, with its pool and the memory it grew to.
        if let Some(epoch) = hal::kosuno_patch::destroy_epoch() {
            let key = &*encoder as *const dyn hal::DynCommandEncoder as *const () as usize;
            let acquired = self.acquired_in.lock().remove(&key);
            if acquired.is_some_and(|a| a < epoch) {
                drop(encoder);
                return;
            }
            self.free_epochs.lock().push(acquired.unwrap_or(epoch));
        }
        let mut free_encoders = self.free_encoders.lock();
        // Kosuno experiment (`hal::kosuno_patch`): past the limit, dropping
        // the encoder destroys its command pool and the memory it held.
        if hal::kosuno_patch::trim_encoders().is_some_and(|n| free_encoders.len() >= n) {
            drop(free_encoders);
            drop(encoder);
            return;
        }
        free_encoders.push(encoder);
        // Kosuno experiment: encoders idle (at the bottom of this LIFO
        // stack) for longer than the age limit go, with their pools.
        if let Some(age) = hal::kosuno_patch::age_ms() {
            let mut released_at = self.released_at.lock();
            let now = std::time::Instant::now();
            released_at.push(now);
            let limit = core::time::Duration::from_millis(age as u64);
            let stale = released_at.iter().take_while(|t| now.duration_since(**t) > limit).count();
            if stale > 0 {
                released_at.drain(..stale);
                let dropped: Vec<_> = free_encoders.drain(..stale).collect();
                drop(free_encoders);
                drop(dropped);
            }
        }
    }
}
