use alloc::alloc::dealloc;
use core::mem::MaybeUninit;
use core::{mem, ptr};
use core::alloc::Layout;
use core::ptr::NonNull;
use core::mem::ManuallyDrop;
use likely_stable::likely;
use crate::HeapArray;

pub(crate) struct Guard<T> {
    pub(crate) ptr: NonNull<MaybeUninit<T>>,
    pub(crate) initialized: usize,
    pub(crate) len: usize
}

impl<T> Guard<T> {
    pub(crate) unsafe fn push_unchecked(&mut self, val: T) {
        self.ptr.cast::<T>().as_ptr().add(self.initialized).write(val);
        self.initialized = self.initialized.wrapping_add(1); // Unchecked add is unstable
    }

    pub(crate) unsafe fn into_heap_array_unchecked(self) -> HeapArray<T> {
        let this = ManuallyDrop::new(self);
        debug_assert!(likely(this.initialized == this.len));
        HeapArray { ptr: this.ptr.cast(), len: this.len }
    }
}

impl<T> Drop for Guard<T> {
    fn drop(&mut self) {
        unsafe {
            ptr::drop_in_place(ptr::slice_from_raw_parts_mut(
                self.ptr.as_ptr() as *mut T, self.initialized
            ));

            debug_assert!(self.len != 0);

            // size is always less than isize::MAX we checked that already
            // By using Layout::array::<T> to allocate
            let size = mem::size_of::<T>().wrapping_mul(self.len);
            let align = mem::align_of::<T>();
            let layout = Layout::from_size_align_unchecked(size, align);
            dealloc(self.ptr.as_ptr().cast(), layout)
        }
    }
}