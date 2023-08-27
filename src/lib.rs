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
//! [`Vec`]: std::vec::Vec
//! [`HeapArray`]: HeapArray

mod try_me;
mod guard;

#[cfg(feature = "serde")]
extern crate serde;
extern crate alloc;

use alloc::{boxed::Box, vec::Vec, alloc::{alloc, dealloc, handle_alloc_error}, vec::IntoIter, vec};
use core::{mem, ptr::{self, NonNull}, fmt::{Debug, Formatter}, mem::{ManuallyDrop, MaybeUninit}, ops::{Deref, DerefMut}, panic::{RefUnwindSafe, UnwindSafe}, alloc::Layout, fmt};
use core::cmp::Ordering;
use core::mem::forget;
use core::ops::ControlFlow;
use core::slice::{Iter, IterMut};
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
/// [`Vec`]: std::vec::Vec
/// [`HeapArray`]: HeapArray
pub struct HeapArray<T> {
    ptr: NonNull<T>,
    len: usize
}
impl<T> HeapArray<T> {
    /// Constructs a new, empty [`HeapArray`] without allocating.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![allow(unused_mut)]
    /// # use heap_array::HeapArray;
    /// let vec: HeapArray<i32> = HeapArray::new();
    /// ```
    /// [`HeapArray`]: HeapArray
    #[inline]
    pub const fn new() -> Self { Self { ptr: NonNull::dangling(), len: 0 } }

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
    pub fn into_boxed_slice(self) -> Box<[T]> { self.into() }

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
    pub fn into_vec(self) -> Vec<T> {
        self.into()
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
    #[inline]
    pub fn from_fn(len: usize, f: impl FnMut(usize) -> T) -> Self {
        Self::try_from_fn(len, NeverShortCircuit::wrap_fn(f))
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
    /// * `f`: function where the passed argument is the current array index.
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
    #[inline]
    pub fn try_from_fn<R>(len: usize, mut f: impl FnMut(usize) -> R) -> R::TryType<Self>
        where R: Try<Output=T>,
    {
        let ptr = match unsafe { alloc_uninnit::<T>(len) } {
            Some(ptr) => ptr,
            None => return R::from_element(Self::new())
        };

        // We use Guard to avoid memory leak in panic's
        let mut guard = Guard { ptr, len, initialized: 0 };
        for i in 0..len {
            match f(i).branch() {
                ControlFlow::Continue(output) => unsafe { guard.push_unchecked(output) }
                ControlFlow::Break(r) => { return R::from_residual(r) }
            }
        }

        // SAFETY: All elements are initialized
        R::from_element(unsafe { guard.into_heap_array_unchecked() })
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
    #[inline]
    pub fn from_element(len: usize, element: T) -> HeapArray<T>
        where T: Clone
    {
        // We use vec![] rather than Self::from_fn(len, |_| element.clone())
        // as it has specialization traits for manny things Such as zero initialization
        // as well as avoid an extra copy (caused by not using element except for cloning)
        vec![element; len].into()
    }
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
    pub fn as_ptr(&self) -> *const T {
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
    pub fn as_slice(&self) -> &[T] {
        self
    }
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
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self
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
    pub fn leak(self) -> &'static mut [T] {
        let mut this = ManuallyDrop::new(self);
        unsafe { core::slice::from_raw_parts_mut(this.as_mut_ptr(), this.len) }
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
    pub fn into_raw_parts(self) -> (NonNull<T>, usize) {
        let this = ManuallyDrop::new(self);
        (this.ptr, this.len)
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
    pub fn from_raw_parts(ptr: NonNull<T>, len: usize) -> Self {
        Self { ptr, len }
    }
    
    
    // Safety: Caller must up hold
    // only call on a Non empty HeapArray
    // Must ensure the HeapArray wont be dropped afterwards
    // and that it wont be accessed later
    pub(crate) unsafe fn dealloc(&mut self) {
        if self.len != 0 {
            // size is always less than isize::MAX we checked that already
            // By using Layout::array::<T> to allocate
            let size = mem::size_of::<T>().wrapping_mul(self.len);
            let align = mem::align_of::<T>();

            let layout = Layout::from_size_align_unchecked(size, align);
            dealloc(self.ptr.as_ptr().cast(), layout)
        }
    }
}
impl<T> Default for HeapArray<T> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone> From<&[T]> for HeapArray<T> {
    fn from(values: &[T]) -> Self {
        HeapArray::from_fn(values.len(), |i| unsafe {
            // Safety: from_fn provides values 0..len
            // and all values gotten should be within that range
            match cfg!(debug_asserions) {
                true => {values.get(i).expect("HeapArray cloning out of bounds").clone()}
                false => {values.get_unchecked(i).clone()}
            }
        })
    }
}
impl<T, const N: usize> From<[T; N]> for HeapArray<T> {
    fn from(values: [T; N]) -> Self {
        let mut ptr = match unsafe { alloc_uninnit::<T>(N) } {
            Some(ptr) => ptr,
            None => return Self::new()
        };

        unsafe { ptr::copy_nonoverlapping(values.as_ptr(), ptr.as_mut().as_mut_ptr(), N) }
        forget(values);

        Self { ptr: ptr.cast(), len: N }
    }
}

impl<T> From<Box<[T]>> for HeapArray<T> {
    #[inline]
    fn from(value: Box<[T]>) -> Self {
        // A box pointer will be a properly aligned and non-null pointer.
        let parts = Box::into_raw(value);
        unsafe { Self { ptr: NonNull::new_unchecked(parts.cast()), len: (*parts).len() } }
    }
}
impl<T> From<Vec<T>> for HeapArray<T> {
    #[inline]
    fn from(value: Vec<T>) -> Self {
        Self::from(value.into_boxed_slice())
    }
}

impl<T> From<HeapArray<T>> for Vec<T> {
    #[inline]
    fn from(value: HeapArray<T>) -> Self {
        let value = ManuallyDrop::new(value);
        unsafe { Vec::from_raw_parts(value.ptr.as_ptr(), value.len, value.len) }
    }
}
impl<T> From<HeapArray<T>> for Box<[T]> {
    #[inline]
    fn from(value: HeapArray<T>) -> Self {
        let this = ManuallyDrop::new(value);
        let ptr = ptr::slice_from_raw_parts_mut(this.ptr.as_ptr(), this.len);
        unsafe { Box::from_raw(ptr) }
    }
}


impl<T, const N: usize> TryFrom<HeapArray<T>> for [T; N] {
    type Error = HeapArray<T>;

    fn try_from(value: HeapArray<T>) -> Result<Self, Self::Error> {
        if value.len != N {
            return Err(value)
        }

        if N == 0 {
            // Wont Panic N is 0, so teh fn wont run
            return Ok(core::array::from_fn(|_| unreachable!()))
        }

        let mut value = ManuallyDrop::new(value);
        let data = unsafe { ptr::read(value.as_ptr() as *const [T; N]) };
        // Safety: value is a ManuallyDrop and so it wort be dropped
        // and since we take ownership of value it wont be accessed after this
        unsafe { value.dealloc(); }

        Ok(data)
    }
}

impl<T, const N: usize> TryFrom<HeapArray<T>> for Box<[T; N]> {
    type Error = HeapArray<T>;

    fn try_from(value: HeapArray<T>) -> Result<Self, Self::Error> {
        if value.len != N {
            Err(value)
        } else {
            let value = ManuallyDrop::new(value);
            // SAFETY: we literally just checked if value.len != N
            Ok(unsafe { Box::from_raw(value.ptr.as_ptr() as *mut [T; N]) })
        }
    }
}

impl<T> Deref for HeapArray<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        unsafe { core::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}
impl<T> DerefMut for HeapArray<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { core::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}
impl<T> AsRef<[T]> for HeapArray<T> {
    fn as_ref(&self) -> &[T] {
        self
    }
}
impl<T> AsMut<[T]> for HeapArray<T> {
    fn as_mut(&mut self) -> &mut [T] {
        self
    }
}
impl<T: Debug> Debug for HeapArray<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        self.deref().fmt(f)
    }
}
impl<T: Clone> Clone for HeapArray<T> {
    fn clone(&self) -> Self {
        HeapArray::from(self.deref())
    }
}



impl<T: PartialEq> PartialEq for HeapArray<T> {
    fn eq(&self, other: &Self) -> bool { self.deref().eq(other.deref()) }
}
impl<T: Eq> Eq for HeapArray<T>{}

impl<T: PartialOrd> PartialOrd for HeapArray<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.deref().partial_cmp(other)
    }
}
impl<T: Ord> Ord for HeapArray<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.deref().cmp(other)
    }
}


#[allow(missing_docs)]
impl<T> Drop for HeapArray<T> {
    fn drop(&mut self) {
        unsafe {
            ptr::drop_in_place(ptr::slice_from_raw_parts_mut(self.ptr.as_ptr(), self.len));

            if !self.is_empty() {
                self.dealloc()
            }
        }
    }
}


// Safety: the pointer we hold is unique to us and so send data is ok to be sent
// and sync data is ok to be synced same goes for Unpin, RefUnwindSafe and UnwindSafe
unsafe impl<T: Send> Send for HeapArray<T>{}
unsafe impl<T: Sync> Sync for HeapArray<T>{}

impl<T: Unpin> Unpin for HeapArray<T>{}
impl<T: RefUnwindSafe> RefUnwindSafe for HeapArray<T>{}
impl<T: UnwindSafe> UnwindSafe for HeapArray<T>{}

// vec dependent impls
#[allow(missing_docs)]
impl<T> FromIterator<T> for HeapArray<T> {
    fn from_iter<I: IntoIterator<Item=T>>(iter: I) -> Self {
        Vec::from_iter(iter).into()
    }
}

#[allow(missing_docs)]
impl<T> IntoIterator for HeapArray<T> {
    type Item = T;
    type IntoIter = IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.into_vec().into_iter()
    }
}

#[allow(missing_docs)]
impl<'a, T> IntoIterator for &'a HeapArray<T> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[allow(missing_docs)]
impl<'a, T> IntoIterator for &'a mut HeapArray<T> {
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

#[allow(missing_docs)]
#[doc(hidden)]
#[inline(always)]
unsafe fn alloc_uninnit<T>(len: usize) -> Option<NonNull<MaybeUninit<T>>> {
    if len == 0 {
        return None
    }

    // We avoid `unwrap_or_else` here because it bloats the LLVM IR generated
    let layout = match Layout::array::<T>(len) {
        Ok(layout) => layout,
        Err(_) => panic!("capacity overflow"),
    };

    if usize::BITS < 64 && unlikely(layout.size() > isize::MAX as usize) {
        panic!("capacity overflow")
    }

    Some(unsafe {
        match NonNull::new(alloc(layout)) {
            Some(ptr) => ptr.cast(),
            None => handle_alloc_error(layout)
        }
    })
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
impl<'de, T: Deserialize<'de>> Deserialize<'de> for HeapArray<T> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error> where D: Deserializer<'de> {
        use core::marker::PhantomData;

        struct HeapArrayVisitor<T> {
            marker: PhantomData<T>,
        }

        #[repr(transparent)]
        struct ExpectedLen(usize);

        impl Expected for ExpectedLen {
            #[inline]
            fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
                formatter.write_str("a length of ")?;
                fmt::Display::fmt(&self.0, formatter)
            }
        }

        impl<'a, T: Deserialize<'a>> Visitor<'a> for HeapArrayVisitor<T> {
            type Value = HeapArray<T>;

            fn expecting(&self, formatter: &mut Formatter) -> fmt::Result {
                formatter.write_str("a sequence")
            }

            fn visit_seq<Arr>(self, mut seq: Arr) -> Result<Self::Value, Arr::Error>
                where
                    Arr: SeqAccess<'a>,
            {
                if let Some(len) = seq.size_hint() {
                    HeapArray::try_from_fn(len, |i| {
                        seq.next_element::<T>().and_then(|res| match res {
                            Some(out) => Ok(out),
                            None => Err(Error::invalid_length(i+1, &ExpectedLen(len)))
                        })
                    })
                } else {
                    let mut values = Vec::<T>::new();
                    while let Some(value) = seq.next_element()? {
                        values.push(value);
                    }

                    Ok(HeapArray::from(values))
                }
            }
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
impl<T: SimdSerialize> SimdSerialize for HeapArray<T> {
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

#[cfg(feature = "simd-json")]
impl<'input, T: SimdDeserialize> SimdDeserialize for HeapArray<T> {
    fn from_tape(tape: &mut Tape<'input>) -> simd_json::Result<Self> where Self: Sized + 'input {
        if let Some(simd_json::Node::Array(size, _)) = tape.next() {
            HeapArray::try_from_fn(size, |_| T::from_tape(tape))
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
    () => { $crate::HeapArray::new() };
    ($($x:expr),+) => { $crate::HeapArray::from([$($x),+]) };
    ($elem:expr; 0) => {{$elem; $crate::HeapArray::new()}};
    ($elem:expr; $n:expr) => { $crate::HeapArray::from_element($n, $elem) };
}