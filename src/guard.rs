use core::mem::MaybeUninit;
use core::{mem, ptr};
use core::ptr::NonNull;

#[cfg(feature = "allocator-api")]
use alloc::alloc::Allocator;
use core::alloc::Layout;
use crate::{assume, HeapArray};

macro_rules! guard_impl {
    (
        pub(crate) unsafe fn into_heap_array_unchecked($($params: tt),*) -> $ret: ty $r#impl: block
    ) => {
        pub(crate) unsafe fn push_unchecked(&mut self, val: T) {
            self.ptr.cast::<T>().as_ptr().add(self.initialized).write(val);
            // Unchecked add is unstable, update when stable
            self.initialized += 1;
        }
        pub(crate) unsafe fn into_heap_array_unchecked($($params),*) -> $ret {
            $r#impl
        }
    };
}

#[cfg(feature = "allocator-api")]
pub(crate) struct Guard<T, A: Allocator> {
    pub(crate) ptr: NonNull<MaybeUninit<T>>,
    pub(crate) len: usize,

    pub(crate) alloc: A,

    pub(crate) initialized: usize
}
#[cfg(not(feature = "allocator-api"))]
pub(crate) struct Guard<T> {
    pub(crate) ptr: NonNull<MaybeUninit<T>>,
    pub(crate) len: usize,

    pub(crate) initialized: usize
}

#[cfg(feature = "allocator-api")]
impl<T, A: Allocator> Guard<T, A> {
    guard_impl! {
        pub(crate) unsafe fn into_heap_array_unchecked(self) -> HeapArray<T, A> {
            let this = mem::ManuallyDrop::new(self);
            unsafe { assume!(this.initialized == this.len) };
            HeapArray::from_raw_parts_in(this.ptr.cast(), this.len, ptr::read(&this.alloc))
        }
    }
}

#[cfg(not(feature = "allocator-api"))]
impl<T> Guard<T> {
    guard_impl! {
        pub(crate) unsafe fn into_heap_array_unchecked(self) -> HeapArray<T> {
            let this = mem::ManuallyDrop::new(self);
            unsafe { assume!(this.initialized == this.len) };
            HeapArray::from_raw_parts(this.ptr.cast(), this.len)
        }
    }
}

macro_rules! drop_impl {
    () => {
        fn drop(&mut self) {
            if mem::needs_drop::<T>() {
                unsafe {
                    ptr::drop_in_place(ptr::slice_from_raw_parts_mut(
                        self.ptr.as_ptr() as *mut T,
                        self.initialized,
                    ));
                }
            }
            unsafe { assume!(self.len != 0) };
            // size is always less than isize::MAX we checked that already
            // By using Layout::array::<T> to allocate

            // But.. unchecked mul is unstable, update when stable
            let size = mem::size_of::<T>() * self.len;
            let align = mem::align_of::<T>();

            unsafe {
                let layout = Layout::from_size_align_unchecked(size, align);
                #[cfg(feature = "allocator-api")]
                self.alloc.deallocate(self.ptr.cast(), layout);
                #[cfg(not(feature = "allocator-api"))]
                alloc::alloc::dealloc(self.ptr.as_ptr() as *mut u8, layout);
            }
        }
    };
}


#[cfg(not(feature = "allocator-api"))]
impl<T> Drop for Guard<T> {
    drop_impl! {}
}


#[cfg(feature = "allocator-api")]
impl<T, A: Allocator> Drop for Guard<T, A> {
    drop_impl! {}
}