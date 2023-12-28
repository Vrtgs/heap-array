# HeapArray
An Implementation of a variable length array, with its main benefit over `Vec` is taking up less space
as `HeapArray` is represented as (pointer, len) while Vec is a (pointer, len, capacity)
and is meant as a replacement for `Box<[T]>`

nice to have: compatible with serde

# Example
```rust
use heap_array::{heap_array, HeapArray};

fn main() {
    let arr = heap_array![1, 2, 5, 8];

    assert_eq!(arr[0], 1);
    assert_eq!(arr[1], 2);
    assert_eq!(arr[2], 5);
    assert_eq!(arr[3], 8);
    assert_eq!(arr.len(), 4);

    let arr = HeapArray::from_fn(10, |i| i);
    assert_eq!(*arr, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
}
```