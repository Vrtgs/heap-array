#![cfg_attr(feature = "allocator-api", feature(allocator_api))]
#![cfg_attr(feature = "dropck", feature(dropck_eyepatch))]

#![warn(missing_docs)]
#![no_std]

//! # HeapArray
//! An Implementation of a variable length array, with its main benefit over [`Vec`] is taking up less space
//! as [`HeapArray`] is represented as (pointer, len) while [`Vec`] is a (pointer, len, capacity)
//! and is meant as a replacement for `Box<[T]>`
//!
//! nice to have: compatible with serde
//!
//! # Examples
//! ```
//! use heap_array::{heap_array, HeapArray};
//! let arr = heap_array![1, 2, 5, 8];
//!
//! assert_eq!(arr[0], 1);
//! assert_eq!(arr[1], 2);
//! assert_eq!(arr[2], 5);
//! assert_eq!(arr[3], 8);
//! assert_eq!(arr.len(), 4);
//!
//! let arr = HeapArray::from_fn(10, |i| i);
//! assert_eq!(*arr, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
//! ```
//!
//! [`Vec`]: Vec
//! [`HeapArray`]: HeapArray

mod try_me;
mod guard;

#[cfg(feature = "serde")]
extern crate serde;
extern crate alloc;

use core::{
    ptr::{self, NonNull},
    fmt::{Debug, Formatter},
    mem::{self, ManuallyDrop, MaybeUninit, forget},
    ops::{Deref, DerefMut, ControlFlow},
    cmp::Ordering,
    slice::{Iter, IterMut},
    marker::PhantomData,
    panic::{RefUnwindSafe, UnwindSafe},
    alloc::Layout,
    fmt
};

use alloc::{
    boxed::Box,
    vec::Vec,
    alloc::{handle_alloc_error},
    vec::IntoIter,
    vec
};

#[cfg(feature = "allocator-api")]
use alloc::alloc::{Allocator, Global};
use core::slice::SliceIndex;
use core::ops::{Index, IndexMut};
use core::hash::{Hash, Hasher};

use likely_stable::{unlikely};
use crate::guard::Guard;
use crate::try_me::{NeverShortCircuit, Try};

/// # HeapArray
/// An Implementation of a variable length array, with its main benefit over [`Vec`] is taking up less space
/// as [`HeapArray`] is represented as (pointer, len) while [`Vec`] is a (pointer, len, capacity)
/// and is meant as a replacement for `Box<[T]>`
///
/// nice to have: compatible with serde
///
/// # Examples
/// ```
/// use heap_array::{heap_array, HeapArray};
/// let arr = heap_array![1, 2, 5, 8];
///
/// assert_eq!(arr[0], 1);
/// assert_eq!(arr[1], 2);
/// assert_eq!(arr[2], 5);
/// assert_eq!(arr[3], 8);
/// assert_eq!(arr.len(), 4);
///
/// let arr = HeapArray::from_fn(10, |i| i);
/// assert_eq!(*arr, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
/// ```
///
/// [`Vec`]: Vec
/// [`HeapArray`]: HeapArray

#[cfg(not(feature = "allocator-api"))]
pub struct HeapArray<T> {
    ptr: NonNull<T>,
    len: usize,

    // NOTE: this marker has no consequences for variance, but is necessary
    // for dropck to understand that we logically own a `T`.
    //
    // For details, see:
    // https://github.com/rust-lang/rfcs/blob/master/text/0769-sound-generic-drop.md#phantom-data
    __marker: PhantomData<T>
}

/// # HeapArray
/// An Implementation of a variable length array, with its main benefit over [`Vec`] is taking up less space
/// as [`HeapArray`] is represented as (pointer, len) while [`Vec`] is a (pointer, len, capacity)
/// and is meant as a replacement for `Box<[T]>`
///
/// nice to have: compatible with serde
///
/// # Examples
/// ```
/// use heap_array::{heap_array, HeapArray};
/// let arr = heap_array![1, 2, 5, 8];
///
/// assert_eq!(arr[0], 1);
/// assert_eq!(arr[1], 2);
/// assert_eq!(arr[2], 5);
/// assert_eq!(arr[3], 8);
/// assert_eq!(arr.len(), 4);
///
/// let arr = HeapArray::from_fn(10, |i| i);
/// assert_eq!(*arr, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
/// ```
///
/// [`Vec`]: Vec
/// [`HeapArray`]: HeapArray
#[cfg(feature = "allocator-api")]
pub struct HeapArray<T, #[cfg(feature = "allocator-api")] A: Allocator = Global> {
    ptr: NonNull<T>,
    len: usize,


    // this is manually drop, so we can run the destructor for the allocation but not the [T] we own
    #[cfg(feature = "allocator-api")]
    alloc: ManuallyDrop<A>,

    // NOTE: this marker has no consequences for variance, but is necessary
    // for dropck to understand that we logically own a `T`.
    //
    // For details, see:
    // https://github.com/rust-lang/rfcs/blob/master/text/0769-sound-generic-drop.md#phantom-data
    __marker: PhantomData<T>
}


macro_rules! assume {
    ($cond:expr $(,)?) => {
        match cfg!(debug_assertions) {
            true  => assert!($cond, "bad assumption"),
            false => if !$cond { core::hint::unreachable_unchecked() },
        }
    };
}

pub(crate) use assume;


macro_rules! from_fn_impl {
    ($T: ty, $R: ty, $len:expr, $f: expr $(, $alloc:ident)?) => {{
        let len = $len;
        let ptr: NonNull<MaybeUninit<$T>> = match alloc_uninit(len $(, &$alloc)?) {
            Some(ptr) => ptr,
            None => {
                #[cfg(feature = "allocator-api")]
                return <$R>::from_element(Self::new_in($($alloc)?));

                #[cfg(not(feature = "allocator-api"))]
                return <$R>::from_element(Self::new());
            },
        };

        // We use Guard to avoid memory leak in panic's
        let mut guard = Guard {
            ptr,
            len,
            $(alloc: $alloc,)?
            initialized: 0
        };
        for i in 0..len {
            // Safety: this loop runs from 0..len
            unsafe { assume!(i < len) }
            match $f(i).branch() {
                ControlFlow::Continue(output) => unsafe { guard.push_unchecked(output) }
                ControlFlow::Break(r) => { return <$R>::from_residual(r) }
            }
        }

        // SAFETY: All elements are initialized
        <$R>::from_element(unsafe { guard.into_heap_array_unchecked() })
    }};
}

macro_rules! from_array_impl {
    ($T:ty, $array:ident, $N:ident $(, $alloc:ident)?) => {{
        let mut ptr: NonNull<MaybeUninit<$T>> = match alloc_uninit($N $(, &$alloc)?) {
            Some(ptr) => ptr,
            None => {
                #[cfg(feature = "allocator-api")]
                return Self::new_in($($alloc)?);

                #[cfg(not(feature = "allocator-api"))]
                return Self::new();
            }
        };

        unsafe { ptr::copy_nonoverlapping($array.as_ptr(), ptr.as_mut().as_mut_ptr(), $N) }
        forget($array);

        #[cfg(feature = "allocator-api")]
        return Self::from_raw_parts_in(ptr.cast(), $N, $($alloc)?);

        #[cfg(not(feature = "allocator-api"))]
        return Self::from_raw_parts(ptr.cast(), $N);
    }};
}

macro_rules! impl_heap_array_inner {
    // base case for recursion
    () => {};

    ($(#[$meta_data:meta])* $visibility:vis fn $name:ident $(<{$($generics:tt)*}>)? ($($args:tt)*) -> $ret:ty $(where {$($restrict:tt)*})? { $($body:tt)* } $($rest:tt)*) => {
        $(#[$meta_data])*
        $visibility fn $name $(<$($generics)*>)? ($($args)*) -> $ret $(where $($restrict)*)? {
            $($body)*
        }


        impl_heap_array_inner! {$($rest)*}
    };
    ($(#[$meta_data:meta])* $visibility:vis const fn $name:ident $(<{$($generics:tt)*}>)? ($($args:tt)*) -> $ret:ty $(where {$($restrict:tt)*})? { $($body:tt)* } $($rest:tt)*) => {
        $(#[$meta_data])*
        $visibility const fn $name $(<$($generics)*>)? ($($args)*) -> $ret $(where $($restrict)*)? {
            $($body)*
        }


        impl_heap_array_inner! {$($rest)*}
    };

    ($(#[$meta_data:meta])* $visibility:vis allocator-api const fn $name:ident $(<{$($generics:tt)*}>)? ($($args:tt)*) -> $ret:ty $(where {$($restrict:tt)*})? { $($body:tt)* } $($rest:tt)*) => {
        #[cfg(feature = "allocator-api")]
        $(#[$meta_data])*
        $visibility const fn $name $(<$($generics)*>)? ($($args)*) -> $ret $(where $($restrict)*)? {
            $($body)*
        }


        impl_heap_array_inner! {$($rest)*}
    };
    ($(#[$meta_data:meta])* $visibility:vis allocator-api fn $name:ident $(<{$($generics:tt)*}>)? ($($args:tt)*) -> $ret:ty $(where {$($restrict:tt)*})? { $($body:tt)* } $($rest:tt)*) => {
        #[cfg(feature = "allocator-api")]
        $(#[$meta_data])*
        $visibility fn $name $(<$($generics)*>)? ($($args)*) -> $ret $(where $($restrict)*)? {
            $($body)*
        }


        impl_heap_array_inner! {$($rest)*}
    };

    ($(#[$meta_data:meta])* $visibility:vis not(allocator-api) const fn $name:ident $(<{$($generics:tt)*}>)? ($($args:tt)*) -> $ret:ty $(where {$($restrict:tt)*})? { $($body:tt)* } $($rest:tt)*) => {
        #[cfg(not(feature = "allocator-api"))]
        $(#[$meta_data])*
        $visibility const fn $name $(<$($generics)*>)? ($($args)*) -> $ret $(where $($restrict)*)? {
            $($body)*
        }


        impl_heap_array_inner! {$($rest)*}
    };
    ($(#[$meta_data:meta])* $visibility:vis not(allocator-api) fn $name:ident $(<{$($generics:tt)*}>)? ($($args:tt)*) -> $ret:ty $(where {$($restrict:tt)*})? { $($body:tt)* } $($rest:tt)*) => {
        #[cfg(not(feature = "allocator-api"))]
        $(#[$meta_data])*
        $visibility fn $name $(<$($generics)*>)? ($($args)*) -> $ret $(where $($restrict)*)? {
            $($body)*
        }


        impl_heap_array_inner! {$($rest)*}
    };

    (raw-rest { $($t:tt)* }) => {$($t)*};
}

macro_rules! impl_heap_array {
    (impl<$T:ident, Maybe<$A:ident>> HeapArray { $($rest:tt)* } ) => {
        #[cfg(feature = "allocator-api")]
        impl<$T, $A: Allocator> HeapArray<$T, $A> {
            impl_heap_array_inner! {$($rest)*}
        }

        #[cfg(not(feature = "allocator-api"))]
        impl<$T> HeapArray<$T> {
            impl_heap_array_inner! {$($rest)*}
        }
    };
}

impl_heap_array! {
    impl<T, Maybe<A>> HeapArray {
        /// Constructs a new, empty [`HeapArray`] without allocating, but bound to an allocator.
        /// you can get back the allocator by calling [`into_raw_parts_with_alloc`]
        /// or get a reference to the
        /// # Examples
        ///
        /// ```
        /// use std::alloc::System;
        /// use heap_array::HeapArray;
        /// let vec: HeapArray<i32, System> = HeapArray::new_in(System);
        /// ```
        /// [`HeapArray`]: HeapArray
        /// [`into_raw_parts_with_alloc`]: HeapArray::into_raw_parts_with_alloc
        #[inline]
        pub allocator-api const fn new_in(alloc: A) -> Self {
            Self::from_raw_parts_in(NonNull::<T>::dangling(), 0, alloc)
        }

        /// Returns `true` if the vector contains no elements.
        ///
        /// # Examples
        ///
        /// ```
        /// # use std::any::Any;
        /// # use heap_array::{heap_array, HeapArray};
        /// let av: HeapArray<&dyn Any> = HeapArray::new();
        /// assert!(av.is_empty());
        ///
        /// let av = heap_array![1, 2, 3];
        /// assert!(!av.is_empty());
        /// ```
        #[inline]
        pub const fn is_empty(&self) -> bool { self.len == 0 }

        /// Returns the number of elements in the heap-array, also referred to
        /// as its 'length'.
        ///
        /// # Examples
        ///
        /// ```
        /// # use heap_array::heap_array;
        /// let a = heap_array![1, 2, 3];
        /// assert_eq!(a.len(), 3);
        /// ```
        #[inline]
        pub const fn len(&self) -> usize { self.len }

        /// Returns a reference to the underlying allocator.
        #[inline]
        pub allocator-api fn allocator(&self) -> &A { &self.alloc }

        /// Returns an unsafe mutable pointer to the heap-array's buffer, or a dangling
        /// raw pointer valid for zero sized reads if the heap-array didn't allocate.
        ///
        /// The caller must ensure that the heap-array outlives the pointer this
        /// function returns, or else it will end up pointing to garbage.
        /// making any pointers to it invalid.
        ///
        /// # Examples
        ///
        /// ```
        /// # use heap_array::HeapArray;
        /// let len = 6;
        /// let mut arr: HeapArray<u8> = HeapArray::from_element(len, 0);
        /// let arr_ptr = arr.as_mut_ptr();
        ///
        /// // Change elements via raw pointer writes
        /// for i in 0..len {
        ///     unsafe {*arr_ptr.add(i) = i as u8;}
        /// }
        /// assert_eq!(*arr, [0, 1, 2, 3, 4, 5]);
        /// ```
        #[inline(always)]
        pub fn as_mut_ptr(&mut self) -> *mut T {
            // We shadow the slice method of the same name to avoid going through
            // `deref_mut`, which creates an intermediate reference.
            self.ptr.as_ptr()
        }

        /// Returns a raw pointer to the heap-array's buffer, or a dangling raw pointer
        /// valid for zero sized reads if the heap-array didn't allocate.
        ///
        /// The caller must ensure that the heap-array outlives the pointer this
        /// function returns, or else it will end up pointing to garbage.
        /// Modifying the heap-array may cause its buffer to be reallocated,
        /// which would also make any pointers to it invalid.
        ///
        /// The caller must also ensure that the memory the pointer (non-transitively) points to
        /// is never written to (except inside an `UnsafeCell`) using this pointer or any pointer
        /// derived from it. If you need to mutate the contents of the slice, use [`as_mut_ptr`].
        ///
        /// # Examples
        ///
        /// ```
        /// # use heap_array::heap_array;
        /// let x = heap_array![0, 2, 4, 6, 8];
        /// let x_ptr = x.as_ptr();
        ///
        /// for i in 0..x.len() {
        ///     unsafe {assert_eq!(*x_ptr.add(i), i * 2);}
        /// }
        /// ```
        ///
        /// [`as_mut_ptr`]: HeapArray::as_mut_ptr
        #[inline(always)]
        pub const fn as_ptr(&self) -> *const T {
            // We shadow the slice method of the same name to avoid going through
            // `deref`, which creates an intermediate reference.
            self.ptr.as_ptr()
        }

        /// Extracts a slice containing the entire array.
        ///
        /// Equivalent to `&s[..]`.
        ///
        /// # Examples
        ///
        /// ```
        /// # use heap_array::heap_array;
        /// use std::io::{self, Write};
        /// let buffer = heap_array![1, 2, 3, 5, 8];
        /// io::sink().write(buffer.as_slice()).unwrap();
        /// ```
        #[inline(always)]
        pub fn as_slice(&self) -> &[T] { self }

        /// Extracts a mutable slice of the entire array.
        ///
        /// Equivalent to `&mut s[..]`.
        ///
        /// # Examples
        ///
        /// ```
        /// use std::io::{self, Read};
        /// # use heap_array::heap_array;
        /// let mut buffer = heap_array![0; 3];
        /// io::repeat(0b101).read_exact(buffer.as_mut_slice()).unwrap();
        /// ```
        #[inline(always)]
        pub fn as_mut_slice(&mut self) -> &mut [T] { self }

        /// Consumes and leaks the [`HeapArray`], returning a mutable reference to the contents,
        /// `&'a mut [T]`. Note that the type `T` must outlive the chosen lifetime
        /// `'a`. If the type has only static references, or none at all, then this
        /// may be chosen to be `'static`.
        ///
        /// This function is mainly useful for data that lives for the remainder of
        /// the program's life. Dropping the returned reference will cause a memory
        /// leak.
        ///
        /// # Examples
        ///
        /// Simple usage:
        ///
        /// ```
        /// # use heap_array::heap_array;
        /// let x = heap_array![1, 2, 3];
        /// let static_ref: &'static mut [usize] = x.leak();
        /// static_ref[0] += 1;
        /// assert_eq!(static_ref, &[2, 2, 3]);
        /// ```
        /// [`HeapArray`]: HeapArray
        pub allocator-api fn leak<{'a}>(self) -> &'a mut [T] where {A: 'a} {
            let (ptr, len) = self.into_raw_parts();
            unsafe { core::slice::from_raw_parts_mut(ptr.as_ptr(), len) }
        }

        /// Consumes and leaks the [`HeapArray`], returning a mutable reference to the contents,
        /// `&'a mut [T]`. Note that the type `T` must outlive the chosen lifetime
        /// `'a`. If the type has only static references, or none at all, then this
        /// may be chosen to be `'static`.
        ///
        /// This function is mainly useful for data that lives for the remainder of
        /// the program's life. Dropping the returned reference will cause a memory
        /// leak.
        ///
        /// # Examples
        ///
        /// Simple usage:
        ///
        /// ```
        /// # use heap_array::heap_array;
        /// let x = heap_array![1, 2, 3];
        /// let static_ref: &'static mut [usize] = x.leak();
        /// static_ref[0] += 1;
        /// assert_eq!(static_ref, &[2, 2, 3]);
        /// ```
        /// [`HeapArray`]: HeapArray
        #[inline]
        pub not(allocator-api) fn leak<{'a}>(self) -> &'a mut [T] {
            let (ptr, len) = self.into_raw_parts();
            unsafe { core::slice::from_raw_parts_mut(ptr.as_ptr(), len) }
        }

        /// Decomposes a [`HeapArray`] into its raw components.
        ///
        /// Returns the raw pointer to the underlying data, the length of
        /// the heap-array (in elements) These are the same arguments in the same
        /// order as the arguments to [`from_raw_parts`].
        ///
        /// After calling this function, the caller is responsible for the
        /// memory previously managed by the [`HeapArray`]. The only way to do
        /// this is to convert the raw pointer, length back
        /// into a [`HeapArray`] with the [`from_raw_parts`] function, allowing
        /// the destructor to perform the cleanup.
        ///
        /// # Examples
        ///
        /// ```
        /// # use heap_array::{heap_array, HeapArray};
        /// let v: HeapArray<i32> = heap_array![-1, 0, 1];
        ///
        /// let (ptr, len) = v.into_raw_parts();
        ///
        /// let rebuilt = unsafe {
        ///     // We can now make changes to the components, such as
        ///     // transmuting the raw pointer to a compatible type.
        ///     let ptr = ptr.cast::<u32>();
        ///
        ///     HeapArray::from_raw_parts(ptr, len)
        /// };
        /// assert_eq!(*rebuilt, [4294967295, 0, 1]);
        /// ```
        /// [`from_raw_parts`]: HeapArray::from_raw_parts
        /// [`HeapArray`]: HeapArray
        #[inline]
        pub fn into_raw_parts(self) -> (NonNull<T>, usize) {
            let this = ManuallyDrop::new(self);
            (this.ptr, this.len)
        }

        /// Decomposes a `HeapArray<T>` into its raw components.
        ///
        /// Returns the raw pointer to the underlying data, the length of the vector (in elements),
        /// and the allocator. These are the same arguments in the same order as the arguments to
        /// [`from_raw_parts_in`].
        ///
        /// After calling this function, the caller is responsible for the
        /// memory previously managed by the `HeapArray`. The only way to do
        /// this is to convert the raw pointer, length, and capacity back
        /// into a `HeapArray` with the [`from_raw_parts_in`] function, allowing
        /// the destructor to perform the cleanup.
        ///
        /// [`from_raw_parts_in`]: HeapArray::from_raw_parts_in
        ///
        /// # Examples
        ///
        /// ```
        /// #![feature(allocator_api)]
        ///
        /// use std::alloc::System;
        /// use heap_array::HeapArray;
        ///
        /// let mut v: HeapArray<i32, System> = HeapArray::from_array_in([-1, 0, 1], System);
        ///
        /// let (ptr, len, alloc) = v.into_raw_parts_with_alloc();
        ///
        /// let rebuilt = unsafe {
        ///     // We can now make changes to the components, such as
        ///     // transmuting the raw pointer to a compatible type.
        ///     let ptr = ptr.cast::<u32>();
        ///
        ///     HeapArray::from_raw_parts_in(ptr, len, alloc)
        /// };
        ///
        /// assert_eq!(*rebuilt, [4294967295, 0, 1]);
        /// ```
        #[inline]
        pub allocator-api fn into_raw_parts_with_alloc(self) -> (NonNull<T>, usize, A) {
            let this = ManuallyDrop::new(self);
            // we never use alloc again, and it doesnt get dropped
            (this.ptr, this.len, unsafe { ptr::read(&*this.alloc) })
        }

        /// Composes a [`HeapArray`] from its raw components, with an allocator.
        ///
        /// After calling this function, the [`HeapArray`] is responsible for the
        /// memory management. The only way to get this back and get back
        /// the raw pointer, length and allocator back is with the [`into_raw_parts_with_alloc`] function, granting you
        /// control of the allocation, and allocator again.
        ///
        /// # Examples
        ///
        /// ```
        /// #![feature(allocator_api)]
        ///
        /// use std::alloc::System;
        /// use heap_array::HeapArray;
        ///
        /// let mut v: HeapArray<i32, System> = HeapArray::from_array_in([-1, 0, 1], System);
        ///
        /// let (ptr, len, alloc) = v.into_raw_parts_with_alloc();
        ///
        /// let rebuilt = unsafe {
        ///     // We can now make changes to the components, such as
        ///     // transmuting the raw pointer to a compatible type.
        ///     let ptr = ptr.cast::<u32>();
        ///
        ///     HeapArray::from_raw_parts_in(ptr, len, alloc)
        /// };
        ///
        /// assert_eq!(*rebuilt, [4294967295, 0, 1]);
        /// ```
        /// [`into_raw_parts`]: HeapArray::into_raw_parts_with_alloc
        /// [`HeapArray`]: HeapArray
        #[inline(always)]
        pub allocator-api const fn from_raw_parts_in(ptr: NonNull<T>, len: usize, alloc: A) -> Self {
            Self { ptr, len, __marker: PhantomData, alloc: ManuallyDrop::new(alloc) }
        }

        /// Composes a [`HeapArray`] from its raw components.
        ///
        /// After calling this function, the [`HeapArray`] is responsible for the
        /// memory management. The only way to get this back and get back
        /// the raw pointer and length back is with the [`into_raw_parts`] function, granting you
        /// control of the allocation again.
        ///
        /// # Examples
        ///
        /// ```
        /// # use heap_array::{heap_array, HeapArray};
        /// let v: HeapArray<i32> = heap_array![-1, 0, 1];
        ///
        /// let (ptr, len) = v.into_raw_parts();
        ///
        /// let rebuilt = unsafe {
        ///     // We can now make changes to the components, such as
        ///     // transmuting the raw pointer to a compatible type.
        ///     let ptr = ptr.cast::<u32>();
        ///
        ///     HeapArray::from_raw_parts(ptr, len)
        /// };
        /// assert_eq!(*rebuilt, [4294967295, 0, 1]);
        /// ```
        /// [`into_raw_parts`]: HeapArray::into_raw_parts
        /// [`HeapArray`]: HeapArray
        #[inline(always)]
        pub not(allocator-api) const fn from_raw_parts(ptr: NonNull<T>, len: usize) -> Self {
            Self { ptr, len, __marker: PhantomData }
        }

        /// Converts `self` into a Box<[T]> without clones or allocation.
        ///
        /// The resulting box can be converted back into an [`HeapArray`] via
        /// `Box<[T]>`'s `into()` method or by calling `HeapArray::from(box)`.
        ///
        /// this is usually due to a library requiring `Box<[T]>` to be used
        /// as HeapArray is supposed to be the replacement for `Box<[T]>`
        ///
        /// # Examples
        ///
        /// ```
        /// # use heap_array::{heap_array, HeapArray};
        /// let s: HeapArray<u32> = heap_array![10, 40, 30];
        /// let x = s.into_boxed_slice();
        /// // `s` cannot be used anymore because it has been converted into `x`.
        ///
        /// let y: Box<[u32]> = Box::new([10, 40, 30]);
        /// assert_eq!(x, y);
        /// ```
        #[inline]
        pub allocator-api fn into_boxed_slice(self) -> Box<[T], A> {
            let (ptr, len, alloc) = self.into_raw_parts_with_alloc();
            let ptr = ptr::slice_from_raw_parts_mut(ptr.as_ptr(), len);
            unsafe { Box::from_raw_in(ptr, alloc) }
        }

        /// Converts `self` into a Box<[T]> without clones or allocation.
        ///
        /// The resulting box can be converted back into an [`HeapArray`] via
        /// `Box<[T]>`'s `into()` method or by calling `HeapArray::from(box)`.
        ///
        /// this is usually due to a library requiring `Box<[T]>` to be used
        /// as HeapArray is supposed to be the replacement for `Box<[T]>`
        ///
        /// # Examples
        ///
        /// ```
        /// # use heap_array::{heap_array, HeapArray};
        /// let s: HeapArray<u32> = heap_array![10, 40, 30];
        /// let x = s.into_boxed_slice();
        /// // `s` cannot be used anymore because it has been converted into `x`.
        ///
        /// let y: Box<[u32]> = Box::new([10, 40, 30]);
        /// assert_eq!(x, y);
        /// ```
        #[inline]
        pub not(allocator-api) fn into_boxed_slice(self) -> Box<[T]> {
            let (ptr, len) = self.into_raw_parts();
            let ptr = ptr::slice_from_raw_parts_mut(ptr.as_ptr(), len);
            unsafe { Box::from_raw(ptr) }
        }

        /// Converts `self` into a vector without clones or allocation.
        ///
        /// The resulting vector can be converted back into an [`HeapArray`] via
        /// `Vec<T>`'s `into()` method or by calling `HeapArray::from(vec)`.
        ///
        /// Should only be used if you plan to resizing, other wise use `into_boxed_slice` for a smaller
        /// type
        /// # Examples
        ///
        /// ```
        /// # use heap_array::{heap_array, HeapArray};
        /// let s: HeapArray<u32> = heap_array![10, 40, 30];
        /// let x = s.into_vec();
        /// // `s` cannot be used anymore because it has been converted into `x`.
        ///
        /// assert_eq!(x, vec![10, 40, 30]);
        /// ```
        #[inline]
        pub allocator-api fn into_vec(self) -> Vec<T, A> {
            self.into()
        }

        /// Converts `self` into a vector without clones or allocation.
        ///
        /// The resulting vector can be converted back into an [`HeapArray`] via
        /// `Vec<T>`'s `into()` method or by calling `HeapArray::from(vec)`.
        ///
        /// Should only be used if you plan to resizing, other wise use `into_boxed_slice` for a smaller
        /// type
        /// # Examples
        ///
        /// ```
        /// # use heap_array::{heap_array, HeapArray};
        /// let s: HeapArray<u32> = heap_array![10, 40, 30];
        /// let x = s.into_vec();
        /// // `s` cannot be used anymore because it has been converted into `x`.
        ///
        /// assert_eq!(x, vec![10, 40, 30]);
        /// ```
        #[inline]
        pub not(allocator-api) fn into_vec(self) -> Vec<T> {
            self.into()
        }

        /// Creates a [`HeapArray`], where each element `T` is the returned value from `f`
        /// using that element's index.
        ///
        /// # Arguments
        ///
        /// * `len`: length of the array.
        /// * `f`: function where the passed argument is the current array index.
        /// * `alloc`: the allocator used to allocate the heap array
        ///
        /// # Example
        ///
        /// ```rust
        /// #![feature(allocator_api)]
        ///
        /// use std::alloc::{System};
        /// use heap_array::HeapArray;
        /// let array: HeapArray<i32, System> = HeapArray::from_fn_in(5, |i| i as i32, System);
        /// // indexes are:     0  1  2  3  4
        /// assert_eq!(*array, [0, 1, 2, 3, 4]);
        ///
        /// let array2: HeapArray<i32, System> = HeapArray::from_fn_in(8, |i| i as i32 * 2, System);
        /// // indexes are:     0  1  2  3  4  5   6   7
        /// assert_eq!(*array2, [0, 2, 4, 6, 8, 10, 12, 14]);
        ///
        /// let bool_arr: HeapArray<bool, System> = HeapArray::from_fn_in(5, |i| i % 2 == 0, System);
        /// // indexes are:       0     1      2     3      4
        /// assert_eq!(*bool_arr, [true, false, true, false, true]);
        /// ```
        /// [`HeapArray`]: HeapArray
        #[inline]
        pub allocator-api fn from_fn_in(len: usize, f: impl FnMut(usize) -> T, alloc: A) -> Self {
            Self::try_from_fn_in(len, NeverShortCircuit::wrap_fn(f), alloc)
        }

        /// Creates a [`HeapArray`] in an allocator, where each element `T` is the returned value from `cb`
        /// using that element's index.
        /// Unlike [`from_fn_in`], where the element creation can't fail, this version will return an error
        /// if any element creation was unsuccessful.
        ///
        /// The return type of this function depends on the return type of the closure.
        /// If you return `Result<T, E>` from the closure, you'll get a `Result<HeapArray<T>, E>`.
        /// If you return `Option<T>` from the closure, you'll get an `Option<HeapArray<T>>`.
        /// # Arguments
        ///
        /// * `len`: length of the array.
        /// * `f`: function where the passed argument is the current array index, and it is guaranteed to run with values from 0..`len` in ascending order.
        /// * `alloc`: allocator used to allocate the array
        ///
        /// # Example
        ///
        /// ```rust
        /// use std::alloc::System;
        /// use heap_array::HeapArray;
        /// let array: Result<HeapArray<u8, System>, _> = HeapArray::try_from_fn_in(5, |i| i.try_into(), System);
        /// assert_eq!(array.as_deref(), Ok([0, 1, 2, 3, 4].as_ref()));
        ///
        /// let array: Result<HeapArray<i8, System>, _> = HeapArray::try_from_fn_in(200, |i| i.try_into(), System);
        /// assert!(array.is_err());
        ///
        /// let array: Option<HeapArray<usize, System>> = HeapArray::try_from_fn_in(4, |i| i.checked_add(100), System);
        /// assert_eq!(array.as_deref(), Some([100, 101, 102, 103].as_ref()));
        ///
        /// let array: Option<HeapArray<usize, System>> = HeapArray::try_from_fn_in(4, |i| i.checked_sub(100), System);
        /// assert_eq!(array, None);
        /// ```
        /// [`HeapArray`]: HeapArray
        /// [`from_fn`]: HeapArray::from_fn
        pub allocator-api fn try_from_fn_in<{R}>(
            len: usize,
            mut f: impl FnMut(usize) -> R,
            alloc: A
        ) -> R::TryType<Self> where {R: Try<Output=T>}
        { from_fn_impl!(T, R, len, f, alloc) }

        /// Create a [`HeapArray`] from a given element and size.
        ///
        ///
        /// This is a specialization of `HeapArray::from_fn(len, |_| element.clone())`
        /// and may result in a minor speed up over it.
        ///
        /// This will use `clone` to duplicate an expression, so one should be careful
        /// using this with types having a nonstandard `Clone` implementation. For
        /// example, `HeapArray::from_element(5, Rc::new(1))` will create a heap-array of five references
        /// to the same boxed integer value, not five references pointing to independently
        /// boxed integers.
        ///
        /// Also, note that `HeapArray::from_element(0, expr)` is allowed, and produces an empty HeapArray.
        /// This will still evaluate `expr`, however, and immediately drop the resulting value, so
        /// be mindful of side effects.
        ///
        /// # Example
        ///
        /// ```rust
        /// # use heap_array::HeapArray;
        /// let array: HeapArray<u8> = HeapArray::from_element(5, 68);
        /// assert_eq!(*array, [68, 68, 68, 68, 68])
        /// ```
        /// [`HeapArray`]: HeapArray
        /// [`heap_array`]: heap_array
        #[inline(always)]
        pub allocator-api fn from_element_in(len: usize, element: T, alloc: A) -> Self
            where {T: Clone}
        {
            // We use vec![] rather than Self::from_fn(len, |_| element.clone())
            // as it has specialization traits for manny things Such as zero initialization
            // as well as avoid an extra copy (caused by not using element except for cloning)
            vec::from_elem_in(element, len, alloc).into()
        }
        /// Copies a slice into a new `HeapArray` with an allocator.
        ///
        /// # Examples
        ///
        /// ```
        /// #![feature(allocator_api)]
        ///
        /// use std::alloc::System;
        /// use heap_array::HeapArray;
        ///
        /// let s = [10, 40, 30];
        /// let x = HeapArray::from_slice_in(&s, System);
        /// // Here, `s` and `x` can be modified independently.
        /// ```
        pub allocator-api fn from_slice_in(slice: &[T], alloc: A) -> HeapArray<T, A>
            where {T: Clone}
        {
            // HeapArray::from_fn_in(slice.len(), |i| {
            //     // Safety: from_fn provides values 0..len
            //     // and all values gotten should be within that range
            //     match cfg!(debug_assertions) {
            //         true => slice.get(i).expect("heapArray cloning out of bounds").clone(),
            //         false => unsafe { slice.get_unchecked(i).clone() }
            //     }
            // }, alloc)


            // we use this for specialization on Copy T
            <[T]>::to_vec_in(slice, alloc).into()
        }

        /// Allocate a `HeapArray<T>` with a custom allocator and move the array's items into it.
        ///
        /// # Examples
        ///
        /// ```
        /// use heap_array::{heap_array, HeapArray};
        ///
        /// assert_eq!(HeapArray::from_array([1, 2, 3]), heap_array![1, 2, 3]);
        /// ```
        pub allocator-api fn from_array_in<{const N: usize}>(array: [T; N], alloc: A) -> HeapArray<T, A> {
            from_array_impl!(T, array, N, alloc)
        }

        /// Constructs a new, empty [`HeapArray`] without allocating.
        ///
        /// # Examples
        ///
        /// ```
        /// # use heap_array::HeapArray;
        /// let vec: HeapArray<i32> = HeapArray::new();
        /// ```
        /// [`HeapArray`]: HeapArray
        #[inline(always)]
        pub not(allocator-api) const fn new() -> Self {
            #[cfg(not(feature = "allocator-api"))]
            return HeapArray::from_raw_parts(NonNull::<T>::dangling(), 0);
        }

        /// Creates a [`HeapArray`], where each element `T` is the returned value from `cb`
        /// using that element's index.
        ///
        /// # Arguments
        ///
        /// * `len`: length of the array.
        /// * `f`: function where the passed argument is the current array index.
        ///
        /// # Example
        ///
        /// ```rust
        /// # use heap_array::HeapArray;
        /// let array = HeapArray::from_fn(5, |i| i);
        /// // indexes are:     0  1  2  3  4
        /// assert_eq!(*array, [0, 1, 2, 3, 4]);
        ///
        /// let array2 = HeapArray::from_fn(8, |i| i * 2);
        /// // indexes are:     0  1  2  3  4  5   6   7
        /// assert_eq!(*array2, [0, 2, 4, 6, 8, 10, 12, 14]);
        ///
        /// let bool_arr = HeapArray::from_fn(5, |i| i % 2 == 0);
        /// // indexes are:       0     1      2     3      4
        /// assert_eq!(*bool_arr, [true, false, true, false, true]);
        /// ```
        /// [`HeapArray`]: HeapArray
        #[inline(always)]
        pub not(allocator-api) fn from_fn(len: usize, f: impl FnMut(usize) -> T) -> Self {
            HeapArray::try_from_fn(len, NeverShortCircuit::wrap_fn(f))
        }

        /// Creates a [`HeapArray`], where each element `T` is the returned value from `cb`
        /// using that element's index.
        /// Unlike [`from_fn`], where the element creation can't fail, this version will return an error
        /// if any element creation was unsuccessful.
        ///
        /// The return type of this function depends on the return type of the closure.
        /// If you return `Result<T, E>` from the closure, you'll get a `Result<HeapArray<T>, E>`.
        /// If you return `Option<T>` from the closure, you'll get an `Option<HeapArray<T>>`.
        /// # Arguments
        ///
        /// * `len`: length of the array.
        /// * `f`: function where the passed argument is the current array index, and it is guaranteed to run with values from 0..`len` in ascending order.
        ///
        /// # Example
        ///
        /// ```rust
        /// # use heap_array::HeapArray;
        /// let array: Result<HeapArray<u8>, _> = HeapArray::try_from_fn(5, |i| i.try_into());
        /// assert_eq!(array.as_deref(), Ok([0, 1, 2, 3, 4].as_ref()));
        ///
        /// let array: Result<HeapArray<i8>, _> = HeapArray::try_from_fn(200, |i| i.try_into());
        /// assert!(array.is_err());
        ///
        /// let array: Option<HeapArray<usize>> = HeapArray::try_from_fn(4, |i| i.checked_add(100));
        /// assert_eq!(array.as_deref(), Some([100, 101, 102, 103].as_ref()));
        ///
        /// let array: Option<HeapArray<usize>> = HeapArray::try_from_fn(4, |i| i.checked_sub(100));
        /// assert_eq!(array, None);
        /// ```
        /// [`HeapArray`]: HeapArray
        /// [`from_fn`]: HeapArray::from_fn
        pub not(allocator-api) fn try_from_fn<{R}>(len: usize, mut f: impl FnMut(usize) -> R) -> R::TryType<Self>
            where {R: Try<Output=T>}
        {
            from_fn_impl!(T, R, len, f)
        }

        /// Create a [`HeapArray`] from a given element and size.
        ///
        ///
        /// This is a specialization of `HeapArray::from_fn(len, |_| element.clone())`
        /// and may result in a minor speed up over it.
        ///
        /// This will use `clone` to duplicate an expression, so one should be careful
        /// using this with types having a nonstandard `Clone` implementation. For
        /// example, `Self::from_element(5, Rc::new(1))` will create a heap-array of five references
        /// to the same boxed integer value, not five references pointing to independently
        /// boxed integers.
        ///
        /// Also, note that `Self::from_element(0, expr)` is allowed, and produces an empty HeapArray.
        /// This will still evaluate `expr`, however, and immediately drop the resulting value, so
        /// be mindful of side effects.
        ///
        /// Also, note `Self::from_element(n, expr)` can always be replaced with `heap_array![expr; n]`
        /// # Example
        ///
        /// ```rust
        /// # use heap_array::HeapArray;
        /// let array: HeapArray<u8> = HeapArray::from_element(5, 68);
        /// assert_eq!(*array, [68, 68, 68, 68, 68])
        /// ```
        /// [`HeapArray`]: HeapArray
        /// [`heap_array`]: heap_array
        #[inline(always)]
        pub not(allocator-api) fn from_element(len: usize, element: T) -> Self
            where {T: Clone}
        {
            // We use vec![] rather than Self::from_fn(len, |_| element.clone())
            // as it has specialization traits for manny things Such as zero initialization
            // as well as avoid an extra copy (caused by not using element except for cloning)
            vec::from_elem(element, len).into()
        }

        /// Copies a slice into a new `HeapArray`.
        ///
        /// # Examples
        ///
        /// ```
        /// #![feature(allocator_api)]
        ///
        /// use std::alloc::System;
        /// use heap_array::HeapArray;
        ///
        /// let s = [10, 40, 30];
        /// let x = HeapArray::from_slice(&s);
        /// // Here, `s` and `x` can be modified independently.
        /// ```
        #[inline(always)]
        pub not(allocator-api) fn from_slice(slice: &[T]) -> Self
            where {T: Clone}
        {
            // we use this for specialization on Copy T
            <[T]>::to_vec(slice).into()
        }

        /// Allocate a `HeapArray<T>` and move the array's items into it.
        ///
        /// # Examples
        ///
        /// ```
        /// use heap_array::{heap_array, HeapArray};
        ///
        /// assert_eq!(HeapArray::from_array([1, 2, 3]), heap_array![1, 2, 3]);
        /// ```
        #[inline(always)]
        pub not(allocator-api) fn from_array<{const N: usize}>(array: [T; N]) -> Self {
            from_array_impl!(T, array, N)
        }

        raw-rest {
            /// Safety: Caller must up hold
            /// Must ensure the HeapArray wont be dropped afterwards
            /// and that it wont be accessed later
            #[inline]
            pub(crate) unsafe fn drop_memory(&mut self) {
                if self.len != 0 {
                    // size is always less than isize::MAX we checked that already
                    // By using Layout::array::<T> to allocate
                    // but unchecked math is unstable
                    // change when stable
                    let size = mem::size_of::<T>() * self.len;
                    let align = mem::align_of::<T>();

                    let layout = Layout::from_size_align_unchecked(size, align);
                    #[cfg(feature = "allocator-api")]
                    self.alloc.deallocate(self.ptr.cast(), layout);
                    #[cfg(not(feature = "allocator-api"))]
                    alloc::alloc::dealloc(self.ptr.as_ptr() as *mut u8, layout);
                }
                #[cfg(feature = "allocator-api")]
                if mem::needs_drop::<A>() { unsafe { ManuallyDrop::drop(&mut self.alloc) } }
            }
        }
    }
}

#[cfg(feature = "allocator-api")]
impl<T> HeapArray<T, Global> {
    /// Copies a slice into a new `HeapArray`.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(allocator_api)]
    ///
    /// use std::alloc::System;
    /// use heap_array::HeapArray;
    ///
    /// let s = [10, 40, 30];
    /// let x = HeapArray::from_slice(&s);
    /// // Here, `s` and `x` can be modified independently.
    /// ```
    #[inline(always)]
    pub fn from_slice(slice: &[T]) -> Self
        where T: Clone
    { HeapArray::<T, Global>::from_slice_in(slice, Global) }

    /// Create a [`HeapArray`] from a given element and size.
    ///
    ///
    /// This is a specialization of `HeapArray::from_fn(len, |_| element.clone())`
    /// and may result in a minor speed up over it.
    ///
    /// This will use `clone` to duplicate an expression, so one should be careful
    /// using this with types having a nonstandard `Clone` implementation. For
    /// example, `Self::from_element(5, Rc::new(1))` will create a heap-array of five references
    /// to the same boxed integer value, not five references pointing to independently
    /// boxed integers.
    ///
    /// Also, note that `Self::from_element(0, expr)` is allowed, and produces an empty HeapArray.
    /// This will still evaluate `expr`, however, and immediately drop the resulting value, so
    /// be mindful of side effects.
    ///
    /// Also, note `Self::from_element(n, expr)` can always be replaced with `heap_array![expr; n]`
    /// # Example
    ///
    /// ```rust
    /// # use heap_array::HeapArray;
    /// let array: HeapArray<u8> = HeapArray::from_element(5, 68);
    /// assert_eq!(*array, [68, 68, 68, 68, 68])
    /// ```
    /// [`HeapArray`]: HeapArray
    /// [`heap_array`]: heap_array
    #[inline(always)]
    pub fn from_element(len: usize, element: T) -> Self
        where T: Clone
    { HeapArray::<T, Global>::from_element_in(len, element, Global) }

    /// Allocate a `HeapArray<T>` and move the array's items into it.
    ///
    /// # Examples
    ///
    /// ```
    /// use heap_array::{heap_array, HeapArray};
    ///
    /// assert_eq!(HeapArray::from_array([1, 2, 3]), heap_array![1, 2, 3]);
    /// ```
    #[inline(always)]
    pub fn from_array<const N: usize>(array: [T; N]) -> Self
    { HeapArray::<T, Global>::from_array_in(array, Global) }

    /// Creates a [`HeapArray`], where each element `T` is the returned value from `cb`
    /// using that element's index.
    /// Unlike [`from_fn`], where the element creation can't fail, this version will return an error
    /// if any element creation was unsuccessful.
    ///
    /// The return type of this function depends on the return type of the closure.
    /// If you return `Result<T, E>` from the closure, you'll get a `Result<HeapArray<T>, E>`.
    /// If you return `Option<T>` from the closure, you'll get an `Option<HeapArray<T>>`.
    /// # Arguments
    ///
    /// * `len`: length of the array.
    /// * `f`: function where the passed argument is the current array index, and it is guaranteed to run with values from 0..`len` in ascending order.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use heap_array::HeapArray;
    /// let array: Result<HeapArray<u8>, _> = HeapArray::try_from_fn(5, |i| i.try_into());
    /// assert_eq!(array.as_deref(), Ok([0, 1, 2, 3, 4].as_ref()));
    ///
    /// let array: Result<HeapArray<i8>, _> = HeapArray::try_from_fn(200, |i| i.try_into());
    /// assert!(array.is_err());
    ///
    /// let array: Option<HeapArray<usize>> = HeapArray::try_from_fn(4, |i| i.checked_add(100));
    /// assert_eq!(array.as_deref(), Some([100, 101, 102, 103].as_ref()));
    ///
    /// let array: Option<HeapArray<usize>> = HeapArray::try_from_fn(4, |i| i.checked_sub(100));
    /// assert_eq!(array, None);
    /// ```
    /// [`HeapArray`]: HeapArray
    /// [`from_fn`]: HeapArray::from_fn
    #[inline(always)]
    pub  fn try_from_fn<R>(len: usize, f: impl FnMut(usize) -> R) -> R::TryType<Self>
        where R: Try<Output=T>
    { HeapArray::<T, Global>::try_from_fn_in(len, f, Global) }

    /// Creates a [`HeapArray`], where each element `T` is the returned value from `cb`
    /// using that element's index.
    ///
    /// # Arguments
    ///
    /// * `len`: length of the array.
    /// * `f`: function where the passed argument is the current array index.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use heap_array::HeapArray;
    /// let array = HeapArray::from_fn(5, |i| i);
    /// // indexes are:     0  1  2  3  4
    /// assert_eq!(*array, [0, 1, 2, 3, 4]);
    ///
    /// let array2 = HeapArray::from_fn(8, |i| i * 2);
    /// // indexes are:     0  1  2  3  4  5   6   7
    /// assert_eq!(*array2, [0, 2, 4, 6, 8, 10, 12, 14]);
    ///
    /// let bool_arr = HeapArray::from_fn(5, |i| i % 2 == 0);
    /// // indexes are:       0     1      2     3      4
    /// assert_eq!(*bool_arr, [true, false, true, false, true]);
    /// ```
    /// [`HeapArray`]: HeapArray
    #[inline(always)]
    pub fn from_fn(len: usize, f: impl FnMut(usize) -> T) -> Self {
        HeapArray::<T, Global>::from_fn_in(len, f, Global)
    }

    /// Constructs a new, empty [`HeapArray`] without allocating.
    ///
    /// # Examples
    ///
    /// ```
    /// # use heap_array::HeapArray;
    /// let vec: HeapArray<i32> = HeapArray::new();
    /// ```
    /// [`HeapArray`]: HeapArray
    #[inline(always)]
    pub const fn new() -> Self {
        HeapArray::<T, Global>::new_in(Global)
    }

    /// Composes a [`HeapArray`] from its raw components.
    ///
    /// After calling this function, the [`HeapArray`] is responsible for the
    /// memory management. The only way to get this back and get back
    /// the raw pointer and length back is with the [`into_raw_parts`] function, granting you
    /// control of the allocation again.
    ///
    /// # Examples
    ///
    /// ```
    /// # use heap_array::{heap_array, HeapArray};
    /// let v: HeapArray<i32> = heap_array![-1, 0, 1];
    ///
    /// let (ptr, len) = v.into_raw_parts();
    ///
    /// let rebuilt = unsafe {
    ///     // We can now make changes to the components, such as
    ///     // transmuting the raw pointer to a compatible type.
    ///     let ptr = ptr.cast::<u32>();
    ///
    ///     HeapArray::from_raw_parts(ptr, len)
    /// };
    /// assert_eq!(*rebuilt, [4294967295, 0, 1]);
    /// ```
    /// [`into_raw_parts`]: HeapArray::into_raw_parts
    /// [`HeapArray`]: HeapArray
    #[inline(always)]
    pub const fn from_raw_parts(ptr: NonNull<T>, len: usize) -> HeapArray<T> {
        HeapArray::<T, Global>::from_raw_parts_in(ptr, len, Global)
    }
}

macro_rules! identical_impl {
    // base case for recursion
    () => {  };
    (
        impl<$($time: lifetime, )?  T $(: {$($t_restrict:tt)+})?, $({$($generics:tt)*},)? Maybe<A $(: {$( $($alloc_restrict:tt)+ )&+})?>> {$($Trait:tt)+} for HeapArray<T, Maybe<A>> $(where {$($restrict:tt)*})? {
            $($r#impl:tt)*
        }
        $($rest:tt)*
    ) => {
        #[cfg(not(feature = "allocator-api"))]
        impl<$($time, )? T $(: $($t_restrict)+)?, $($($generics)*,)?> $($Trait)+ for HeapArray<T> $(where $($restrict)*)? {
            $($r#impl)*
        }
        #[cfg(feature = "allocator-api")]
        impl<$($time, )? T $(: $($t_restrict)+)?, $($($generics)*,)? A: Allocator $(+ $($($alloc_restrict)*)+)?> $($Trait)+ for HeapArray<T, A> $(where $($restrict)*)? {
            $($r#impl)*
        }

        identical_impl! { $($rest)* }
    };
    (
        unsafe impl<$($time: lifetime, )? T $(: {$($t_restrict:tt)+})?, $({$($generics:tt)*},)? Maybe<A$(: {$( $($alloc_restrict:tt)+ )&+})?>> {$($Trait:tt)+} for HeapArray<T, Maybe<A>> $(where {$($restrict:tt)*})? {
            $($r#impl:tt)*
        }
        $($rest:tt)*
    ) => {
        #[cfg(not(feature = "allocator-api"))]
        unsafe impl<$($time, )? T $(: $($t_restrict)+)?, $($($generics)*,)?> $($Trait)+ for HeapArray<T> $(where $($restrict)*)? {
            $($r#impl)*
        }
        #[cfg(feature = "allocator-api")]
        unsafe impl<$($time, )? T $(: $($t_restrict)+)?, $($($generics)*,)? A: Allocator $(+ $($($alloc_restrict)*)+)?> $($Trait)+ for HeapArray<T, A> $(where $($restrict)*)? {
            $($r#impl)*
        }

        identical_impl! {$($rest)*}
    };
}


impl<T> Default for HeapArray<T> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone> From<&[T]> for HeapArray<T> {
    #[inline(always)]
    fn from(slice: &[T]) -> Self { HeapArray::from_slice(slice) }
}

impl<T, const N: usize> From<[T; N]> for HeapArray<T> {
    fn from(values: [T; N]) -> Self {
        Self::from_array(values)
    }
}

#[cfg(not(feature = "allocator-api"))]
impl<T> From<Box<[T]>> for HeapArray<T> {
    #[inline]
    fn from(value: Box<[T]>) -> Self {
        let raw = Box::into_raw(value);
        unsafe {
            // A box pointer will always be a properly aligned non-null pointer.
            Self::from_raw_parts(NonNull::new_unchecked(raw.cast()), (*raw).len())
        }
    }
}

#[cfg(feature = "allocator-api")]
impl<T, A: Allocator> From<Box<[T], A>> for HeapArray<T, A> {
    #[inline]
    fn from(value: Box<[T], A>) -> Self {
        let (parts, alloc) = Box::into_raw_with_allocator(value);
        unsafe {
            // A box pointer will always be a properly aligned non-null pointer.
            Self::from_raw_parts_in(NonNull::new_unchecked(parts.cast()), (*parts).len(), alloc)
        }
    }
}

#[cfg(not(feature = "allocator-api"))]
impl<T> From<Vec<T>> for HeapArray<T> {
    #[inline]
    fn from(value: Vec<T>) -> Self {
        Self::from(value.into_boxed_slice())
    }
}

#[cfg(feature = "allocator-api")]
impl<T, A: Allocator> From<Vec<T, A>> for HeapArray<T, A> {
    #[inline]
    fn from(value: Vec<T, A>) -> Self {
        Self::from(value.into_boxed_slice())
    }
}



// error[E0210] when trying to do these with alloc-api
impl<T> From<HeapArray<T>> for Box<[T]> {
    #[inline]
    fn from(value: HeapArray<T>) -> Self {
        value.into_boxed_slice()
    }
}

impl<T, const N: usize> TryFrom<HeapArray<T>> for Box<[T; N]> {
    type Error = HeapArray<T>;

    fn try_from(value: HeapArray<T>) -> Result<Self, Self::Error> {
        if value.len != N {
            Err(value)
        } else {
            let value = ManuallyDrop::new(value);
            let ptr = value.ptr;

            // TODO: alloc-api
            // // we never use alloc again, and it doesnt get dropped
            // let alloc = unsafe { ptr::read(&*value.alloc) };

            // SAFETY: we just checked if value.len != N
            let ptr = ptr.as_ptr() as *mut [T; N];
            Ok(unsafe { Box::from_raw(ptr) })
        }
    }
}



#[cfg(not(feature = "allocator-api"))]
impl<T> From<HeapArray<T>> for Vec<T> {
    #[inline]
    fn from(value: HeapArray<T>) -> Self {
        let (ptr, len) = value.into_raw_parts();
        unsafe { Vec::from_raw_parts(ptr.as_ptr(), len, len) }
    }
}

#[cfg(feature = "allocator-api")]
impl<T, A: Allocator> From<HeapArray<T, A>> for Vec<T, A> {
    #[inline]
    fn from(value: HeapArray<T, A>) -> Self {
        let (ptr, len, alloc) = value.into_raw_parts_with_alloc();
        unsafe { Vec::from_raw_parts_in(ptr.as_ptr(), len, len, alloc) }
    }
}


macro_rules! try_from_array_impl {
    ($T: ty, $N: ident, $value: ident) => {{
        if $N == 0 {
            // Safety: N is 0
            return Ok(unsafe { ptr::read(&[] as *const [T; $N]) })
        }

        if $value.len != N {
            return Err($value)
        }

        let mut value = ManuallyDrop::new($value);
        let data: [T; $N] = unsafe { ptr::read(value.as_ptr() as *const [T; $N]) };
        // Safety: value is a ManuallyDrop and so it wont be dropped
        // and since we take ownership of value it wont be accessed after this
        unsafe { value.drop_memory() }

        Ok(data)
    }};
}

#[cfg(feature = "allocator-api")]
impl<T, A: Allocator, const N: usize> TryFrom<HeapArray<T, A>> for [T; N] {
    type Error = HeapArray<T, A>;

    fn try_from(value: HeapArray<T, A>) -> Result<[T; N], Self::Error> {
        try_from_array_impl! (T, N, value)
    }
}

#[cfg(not(feature = "allocator-api"))]
impl<T, const N: usize> TryFrom<HeapArray<T>> for [T; N] {
    type Error = HeapArray<T>;

    fn try_from(value: HeapArray<T>) -> Result<[T; N], Self::Error> {
        try_from_array_impl! (T, N, value)
    }
}


identical_impl! {
    impl<T: {Clone}, Maybe<A: {Clone}>> {Clone} for HeapArray<T, Maybe<A>> {
        fn clone(&self) -> Self {
            #[cfg(not(feature = "allocator-api"))]
            {HeapArray::from_slice(self)}
            #[cfg(feature = "allocator-api")]
            {HeapArray::from_slice_in(self, self.allocator().clone())}
        }
    }

}


identical_impl! {
    impl<T: {Hash}, Maybe<A>> {Hash} for HeapArray<T, Maybe<A>> {
        fn hash<H: Hasher>(&self, state: &mut H) {
            <[T] as Hash>::hash(self.deref(), state)
        }
    }

    impl<T, Maybe<A>> {Deref} for HeapArray<T, Maybe<A>> {
        type Target = [T];

        fn deref(&self) -> &Self::Target {
            unsafe { core::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
        }
    }

    impl<T, Maybe<A>> {DerefMut} for HeapArray<T, Maybe<A>> {
        fn deref_mut(&mut self) -> &mut Self::Target {
            unsafe { core::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
        }
    }

    impl<T, Maybe<A>> {AsRef<[T]>} for HeapArray<T, Maybe<A>> {
        fn as_ref(&self) -> &[T] { self }
    }

    impl<T, Maybe<A>> {AsMut<[T]>} for HeapArray<T, Maybe<A>> {
        fn as_mut(&mut self) -> &mut [T] { self }
    }

    impl<T: {Debug}, Maybe<A>> {Debug} for HeapArray<T, Maybe<A>> {
        fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result { self.deref().fmt(f) }
    }

    impl<T, {I: SliceIndex<[T]>}, Maybe<A>> {Index<I>} for HeapArray<T, Maybe<A>> {
        type Output = I::Output;

        fn index(&self, index: I) -> &Self::Output {
            <[T] as Index<I>>::index(self.deref(), index)
        }
    }

    impl<T, {I: SliceIndex<[T]>}, Maybe<A>> {IndexMut<I>} for HeapArray<T, Maybe<A>> {
        fn index_mut(&mut self, index: I) -> &mut Self::Output {
            <[T] as IndexMut<I>>::index_mut(self.deref_mut(), index)
        }
    }
}


macro_rules! impl_deref_comp_trait {
    ($trait_name: ident |> fn $fn_name:ident(&self, other: &Self) -> $t: ty) => {
        identical_impl! {
            impl<T: {$trait_name}, Maybe<A>> {$trait_name} for HeapArray<T, Maybe<A>> {
                fn $fn_name(&self, other: &Self) -> $t { self.deref().$fn_name(other.deref()) }
            }
        }
    };
}

impl_deref_comp_trait!(PartialEq  |> fn eq(&self, other: &Self) -> bool);
impl_deref_comp_trait!(Ord        |> fn cmp(&self, other: &Self) -> Ordering);
impl_deref_comp_trait!(PartialOrd |> fn partial_cmp(&self, other: &Self) -> Option<Ordering>);
identical_impl! {
    impl<T: {Eq}, Maybe<A>> {Eq} for HeapArray<T, Maybe<A>> {}
}

macro_rules! drop_impl {
    () => {
        fn drop(&mut self) {
            if mem::needs_drop::<T>() {
                unsafe { ptr::drop_in_place(ptr::slice_from_raw_parts_mut(self.as_mut_ptr(), self.len)) };
            }

            unsafe { self.drop_memory() }
        }
    };
}

#[allow(missing_docs)]
#[cfg(all(feature = "dropck", feature = "allocator-api"))]
unsafe impl<#[may_dangle] T, A: Allocator> Drop for HeapArray<T, A> {
    drop_impl! {  }
}

#[allow(missing_docs)]
#[cfg(all(feature = "dropck", not(feature = "allocator-api")))]
unsafe impl<#[may_dangle] T> Drop for HeapArray<T> {
    drop_impl! {  }
}

#[allow(missing_docs)]
#[cfg(all(not(feature = "dropck"), feature = "allocator-api"))]
impl<T, A: Allocator> Drop for HeapArray<T, A> {
    drop_impl! {  }
}

#[allow(missing_docs)]
#[cfg(all(not(feature = "dropck"), not(feature = "allocator-api")))]
impl<T> Drop for HeapArray<T> {
    drop_impl! {  }
}

#[allow(unused)]
fn validate() {
    #[allow(drop_bounds)]
    fn check<T: Drop>() {};
    // can be any type
    check::<HeapArray<u8>>();
}

identical_impl! {
    // Safety: the pointer we hold is unique to us and so send data is ok to be sent
    // and sync data is ok to be synced same goes for Unpin, RefUnwindSafe and UnwindSafe
    unsafe impl<T: {Send}, Maybe<A: {Send}>> {Send} for HeapArray<T, Maybe<A>> {}
    unsafe impl<T: {Sync}, Maybe<A: {Sync}>> {Sync} for HeapArray<T, Maybe<A>> {}

    impl<T: {Unpin}, Maybe<A: {Unpin}>> {Unpin} for HeapArray<T, Maybe<A>>{}
    impl<T: {RefUnwindSafe}, Maybe<A: {RefUnwindSafe}>> {RefUnwindSafe} for HeapArray<T, Maybe<A>> {}
    impl<T: {UnwindSafe}, Maybe<A: {UnwindSafe}>> {UnwindSafe} for HeapArray<T, Maybe<A>> {}
}

// vec dependent impls
#[allow(missing_docs)]
impl<T> FromIterator<T> for HeapArray<T> {
    fn from_iter<I: IntoIterator<Item=T>>(iter: I) -> Self {
        Vec::from_iter(iter).into()
    }
}

#[cfg(feature = "allocator-api")]
#[allow(missing_docs)]
impl<T, A: Allocator> IntoIterator for HeapArray<T, A> {
    type Item = T;
    type IntoIter = IntoIter<Self::Item, A>;

    fn into_iter(self) -> Self::IntoIter {
        self.into_vec().into_iter()
    }
}

#[cfg(not(feature = "allocator-api"))]
#[allow(missing_docs)]
impl<T> IntoIterator for HeapArray<T> {
    type Item = T;
    type IntoIter = IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.into_vec().into_iter()
    }
}

#[cfg(feature = "allocator-api")]
#[allow(missing_docs)]
impl<'a, T, A: Allocator> IntoIterator for &'a HeapArray<T, A> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[cfg(not(feature = "allocator-api"))]
#[allow(missing_docs)]
impl<'a, T> IntoIterator for &'a HeapArray<T> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[cfg(feature = "allocator-api")]
#[allow(missing_docs)]
impl<'a, T, A: Allocator> IntoIterator for &'a mut HeapArray<T, A> {
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T>;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

#[cfg(not(feature = "allocator-api"))]
#[allow(missing_docs)]
impl<'a, T> IntoIterator for &'a mut HeapArray<T> {
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T>;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

#[allow(missing_docs)]
#[doc(hidden)]
#[inline]
fn alloc_uninit<T, #[cfg(feature = "allocator-api")] A: Allocator>(
    len: usize,
    #[cfg(feature = "allocator-api")]
    alloc: &A
) -> Option<NonNull<MaybeUninit<T>>>
{
    if len == 0 {
        return None
    }

    if mem::size_of::<T>() == 0 {
        panic!("ZSTs NOT YET SUPPORTED")
    }

    #[cold]
    const fn capacity_overflow() -> ! {
        panic!("capacity overflow")
    }

    // We avoid `unwrap_or_else` here because it bloats the LLVM IR generated
    let layout = match Layout::array::<T>(len) {
        Ok(layout) => layout,
        Err(_) => capacity_overflow(),
    };

    if usize::BITS < 64 && unlikely(layout.size() > isize::MAX as usize) {
        capacity_overflow()
    }

    #[cfg(feature = "allocator-api")]
    {
        Some(match A::allocate(alloc, layout) {
            // currently allocate returns a [u8] with the same size as the one requested
            // and so we can safely discard its length
            Ok(ptr) => NonNull::<[u8]>::cast::<MaybeUninit<T>>(ptr),
            Err(_) => handle_alloc_error(layout)
        })
    }

    #[cfg(not(feature = "allocator-api"))]
    {
        Some(match NonNull::new(unsafe { alloc::alloc::alloc(layout) }) {
            Some(ptr) => ptr.cast(),
            None => handle_alloc_error(layout)
        })
    }
}


#[cfg(feature = "serde")]
use serde::{
    Deserialize, Deserializer, Serialize, Serializer,
    de::{SeqAccess, Visitor, Error, Expected},
};

#[cfg(feature = "serde")]
impl<T: Serialize> Serialize for HeapArray<T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error> where S: Serializer {
        serializer.collect_seq(self)
    }
}



#[cfg(feature = "serde")]
impl<'a, T: Deserialize<'a>> HeapArray<T> {
    /// Deserializes the heap array from a [`SeqAccess`]
    pub fn from_sequence<A: SeqAccess<'a>>(mut sequence: A) -> Result<Self, A::Error> {
        #[repr(transparent)]
        struct ExpectedLen(usize);

        impl Expected for ExpectedLen {
            #[inline]
            fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
                formatter.write_str("a length of ")?;
                fmt::Display::fmt(&self.0, formatter)
            }
        }

        if let Some(len) = sequence.size_hint() {
            HeapArray::try_from_fn(len, |i| sequence.next_element::<T>().and_then(|res| match res {
                Some(out) => Ok(out),
                None => Err(Error::invalid_length(i+1, &ExpectedLen(len)))
            }))
        } else {
            let mut values = Vec::<T>::new();
            while let Some(value) = sequence.next_element()? {
                values.push(value);
            }

            Ok(HeapArray::from(values))
        }
    }
}


#[cfg(feature = "serde")]
impl<'de, T: Deserialize<'de>> Deserialize<'de> for HeapArray<T> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error> where D: Deserializer<'de> {
        struct HeapArrayVisitor<T> {
            marker: PhantomData<T>,
        }

        impl<'a, T: Deserialize<'a>> Visitor<'a> for HeapArrayVisitor<T> {
            type Value = HeapArray<T>;

            fn expecting(&self, formatter: &mut Formatter) -> fmt::Result {
                formatter.write_str("a sequence")
            }

            #[inline(always)]
            fn visit_seq<A>(self, seq: A) -> Result<Self::Value, A::Error>
                where A: SeqAccess<'a>,
            { HeapArray::from_sequence(seq) }
        }

        let visitor = HeapArrayVisitor::<T> {
            marker: PhantomData,
        };

        deserializer.deserialize_seq(visitor)
    }
}

#[cfg(feature = "simd-json")]
use simd_json_derive::{Serialize as SimdSerialize, Deserialize as SimdDeserialize, Tape};

#[cfg(feature = "simd-json")]
extern crate std;

#[cfg(feature = "simd-json")]
identical_impl! {
    impl<T: {SimdSerialize}, Maybe<A>> {SimdSerialize} for HeapArray<T, Maybe<A>> {
        fn json_write<W>(&self, writer: &mut W) -> simd_json_derive::Result where W: std::io::Write {
            let mut i = self.iter();
            if let Some(first) = i.next() {
                writer.write_all(b"[")?;
                first.json_write(writer)?;
                for e in i {
                    writer.write_all(b",")?;
                    e.json_write(writer)?;
                }
                writer.write_all(b"]")
            } else {
                writer.write_all(b"[]")
            }
        }
    }
}


#[cfg(feature = "simd-json")]
impl<'input, T: SimdDeserialize<'input>> SimdDeserialize<'input> for HeapArray<T>
    where T: 'input
{
    fn from_tape(tape: &mut Tape<'input>) -> simd_json::Result<Self>  {
        if let Some(simd_json::Node::Array { len, .. }) = tape.next() {
            HeapArray::try_from_fn(len, |_| T::from_tape(tape))
        } else {
            Err(simd_json::Error::generic(
                simd_json::ErrorType::ExpectedArray,
            ))
        }
    }
}

/// Creates a [`HeapArray`] containing the arguments.
///
/// `heap_array!` allows `HeapArray`'s to be defined with the same syntax as array expressions.
/// There are two forms of this macro:
///
/// - Create a [`HeapArray`] containing a given list of elements:
///
/// ```
/// # use heap_array::heap_array;
/// let v = heap_array![1, 2, 3];
/// assert_eq!(v[0], 1);
/// assert_eq!(v[1], 2);
/// assert_eq!(v[2], 3);
/// ```
///
/// - Create a [`HeapArray`] from a given element and size:
///
/// ```
/// # use heap_array::heap_array;
/// let v = heap_array![1; 3];
/// assert_eq!(*v, [1, 1, 1]);
/// ```
///
/// Note that unlike array expressions this syntax supports all elements
/// which implement [`Clone`] and the number of elements doesn't have to be
/// a constant.
///
/// This will use `clone` to duplicate an expression, so one should be careful
/// using this with types having a nonstandard `Clone` implementation. For
/// example, `heap_array![Rc::new(1); 5]` will create a heap-array of five references
/// to the same boxed integer value, not five references pointing to independently
/// boxed integers.
///
/// Also, note that `heap_array![expr; 0]` is allowed, and produces an empty HeapArray.
/// This will still evaluate `expr`, however, and immediately drop the resulting value, so
/// be mindful of side effects.
///
/// [`HeapArray`]: HeapArray
#[macro_export]
macro_rules! heap_array {
    [] => { $crate::HeapArray::new() };
    [$($x:expr),+] => { $crate::HeapArray::from([$($x),+]) };
    [$elem:expr; 0] => {{$elem; $crate::HeapArray::new()}};
    [$elem:expr; $n:expr] => { $crate::HeapArray::from_element($n, $elem) };
}