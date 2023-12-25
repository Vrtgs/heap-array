use core::convert::Infallible;
use core::ops::ControlFlow;

type Residual<T> = <T as Try>::TryType<Infallible>;

mod sealed {
    use super::NeverShortCircuit;
    pub trait Sealed {}

    impl<T> Sealed for Option<T> {}
    impl<T, E> Sealed for Result<T, E> {}
    impl<T> Sealed for NeverShortCircuit<T> {}
}

pub trait Try: sealed::Sealed {
    type Output;
    type TryType<T>;

    fn from_element<ELM>(e: ELM) -> Self::TryType<ELM>;
    fn from_residual<T>(residual: Residual<Self>) -> Self::TryType<T>;
    fn branch(self) -> ControlFlow<Residual<Self>, Self::Output>;
}


macro_rules! try_impl {
    (
        $T: ident < T $(, $($generics: tt),* )? >,
        $success:ident ($output:ident),
        $fail: ident $(($err:ident))?
    ) => {
        impl<T $(, $($generics)*)?> Try for $T<T $(, $($generics)*)?> {
            type Output = T;
            type TryType<TT> = $T<TT $(, $($generics)* )?>;

            #[inline(always)]
            fn from_element<ELM>(e: ELM) -> Self::TryType<ELM> {
                $success(e)
            }

            #[inline(always)]
            fn from_residual<TT>(residual: Residual<Self>) -> Self::TryType<TT> {
                match residual {
                    $success(infallible) => match infallible {},
                    $fail$(($err))? => {$fail$(($err))?}
                }
            }

            #[inline(always)]
            fn branch(self) -> ControlFlow<Residual<Self>, Self::Output> {
                match self {
                    $success($output) => ControlFlow::Continue($output),
                    $fail$(($err))? => ControlFlow::Break($fail$(($err))?)
                }
            }
        }
    };
}

try_impl! {
    Option<T>, Some(out), None
}
try_impl! {
    Result<T, E>, Ok(out), Err(err)
}


pub(crate) struct NeverShortCircuit<T>(pub T);

impl<T> NeverShortCircuit<T> {
    #[inline(always)]
    pub fn wrap_fn<Arg>(mut f: impl FnMut(Arg) -> T) -> impl FnMut(Arg) -> NeverShortCircuit<T> {
        move |arg| NeverShortCircuit(f(arg))
    }
}

impl<T> Try for NeverShortCircuit<T> {
    type Output = T;
    type TryType<TT> = TT;

    #[inline(always)]
    fn from_element<ELM>(e: ELM) -> Self::TryType<ELM> { e }

    #[inline(always)]
    fn from_residual<TT>(infallible: Residual<Self>) -> Self::TryType<TT> {
        match infallible {}
    }

    #[inline(always)]
    fn branch(self) -> ControlFlow<Residual<Self>, Self::Output> {
        ControlFlow::Continue(self.0)
    }
}