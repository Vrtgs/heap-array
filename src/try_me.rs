use core::convert::Infallible;
use core::ops::ControlFlow;

pub trait Try {
    type Output;
    type Residual;
    type TryType<T>;

    fn from_element<ELM>(e: ELM) -> Self::TryType<ELM>;
    fn from_residual<T>(residual: Self::Residual) -> Self::TryType<T>;
    fn branch(self) -> ControlFlow<Self::Residual, Self::Output>;
}

impl<T> Try for Option<T> {
    type Output = T;
    type Residual = Option<Infallible>;
    type TryType<TT> = Option<TT>;

    #[inline(always)]
    fn from_element<ELM>(e: ELM) -> Self::TryType<ELM> {
        Some(e)
    }

    #[inline(always)]
    fn from_residual<TT>(residual: Self::Residual) -> Self::TryType<TT> {
        match residual {
            Some(infallible) => match infallible{},
            None => {None}
        }
    }

    #[inline(always)]
    fn branch(self) -> ControlFlow<Self::Residual, Self::Output> {
        match self {
            Some(out) => {ControlFlow::Continue(out)}
            None => {ControlFlow::Break(None)}
        }
    }
}
impl<T, E> Try for Result<T, E> {
    type Output = T;
    type Residual = Result<Infallible, E>;
    type TryType<TT> = Result<TT, E>;

    #[inline(always)]
    fn from_element<ELM>(e: ELM) -> Self::TryType<ELM> {
        Ok(e)
    }

    #[inline(always)]
    fn from_residual<TT>(residual: Self::Residual) -> Self::TryType<TT> {
        match residual {
            Ok(infallible) => match infallible{},
            Err(err) => {Err(err)}
        }
    }

    #[inline(always)]
    fn branch(self) -> ControlFlow<Self::Residual, Self::Output> {
        match self {
            Ok(out) => {ControlFlow::Continue(out)}
            Err(err) => {ControlFlow::Break(Err(err))}
        }
    }
}

#[repr(transparent)]
pub(crate) struct NeverShortCircuit<T>(pub T);

impl<T> NeverShortCircuit<T> {
    #[inline(always)]
    pub fn wrap_fn<Arg>(mut f: impl FnMut(Arg) -> T) -> impl FnMut(Arg) -> NeverShortCircuit<T> {
        move |arg| NeverShortCircuit(f(arg))
    }
}

impl<T> Try for NeverShortCircuit<T> {
    type Output = T;
    type Residual = Infallible;
    type TryType<TT> = TT;

    #[inline(always)]
    fn from_element<ELM>(e: ELM) -> Self::TryType<ELM> {
        e
    }

    #[inline(always)]
    fn from_residual<TT>(infallible: Self::Residual) -> Self::TryType<TT> {
        match infallible{}
    }

    #[inline(always)]
    fn branch(self) -> ControlFlow<Self::Residual, Self::Output> {
        ControlFlow::Continue(self.0)
    }
}