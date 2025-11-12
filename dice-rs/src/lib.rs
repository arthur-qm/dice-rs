pub mod events;
pub mod log;
pub mod raw;

pub type ChainId = u16;
pub type TypeId = u16;
pub type DiceThreadId = u64;
use std::{
    alloc::{GlobalAlloc, Layout},
    marker::PhantomData,
};

#[repr(u16)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Chain {
    InterceptEvent = 1,
    InterceptBefore = 2,
    InterceptAfter = 3,
    CaptureEvent = 4,
    CaptureBefore = 5,
    CaptureAfter = 6,
}

pub trait DiceEvent: Sized {
    const ID: TypeId;

    fn fallback<'a>() -> Option<&'a Self> {
        None
    }

    /// # Safety
    /// This function is intended to be used for casting a void pointer from a dice callback to a Rust reference.
    /// Here we assume that dice correctly returns the pointers to correctly allocated types.
    /// If the Event does not require any data, dice will not provide a type/struct for this and return a nullpointer
    /// The `#[dice_event(...)]` attribute macro correctly handles this and implement a fallback.
    /// The fallback is a &T which is a valid reference for unit structs (structs without a body)
    #[inline]
    unsafe fn from_raw<'a>(ptr: *const ()) -> Option<&'a Self> {
        if ptr.is_null() {
            Self::fallback()
        } else {
            // SAFETY: we know ptr is not null. and from the dice protocol we know this is also correctly typed.
            Some(unsafe { &*(ptr as *const Self) })
        }
    }
}

#[repr(i32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DiceResult {
    Ok = 0,
    StopChain = 1,
    DropEvent = 2,
    HandlerOff = 3,
    Invalid = -1,
    Error = -2,
}

// For now user code is unable to construct this (besides unsafe casting)
#[repr(C, align(8))]
#[derive(Debug)]
pub struct Metadata {
    drop_: bool,
    // This marker makes Metadata !Send and !Sync
    _marker: PhantomData<*mut ()>,
}

pub struct MempoolAllocator;

unsafe impl GlobalAlloc for MempoolAllocator {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: we assume dice allocates correctly
        // we do an additional sanity check in Debug build to verify this.
        let ptr = unsafe { raw::mempool_alloc(layout.size()) as *mut u8 };
        debug_assert!(
            !ptr.is_null() && (ptr as *mut usize).is_aligned(),
            "Sanity Check"
        );
        ptr
    }

    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, _layout: Layout) {
        // SAFETY: we assume that dice is doing the correct thing.
        // from rust persepctive this is a black box
        unsafe { raw::mempool_free(ptr as *mut _) };
    }
}

#[cfg(feature = "dice-self")]
pub mod thread {
    use std::{marker::PhantomData, mem::MaybeUninit};

    use crate::{DiceThreadId, Metadata, raw};

    pub fn self_id(mt: &mut Metadata) -> DiceThreadId {
        // SAFETY: we asssume the extern dice is returning the correct id
        unsafe { raw::thread::self_id(mt) }
    }

    // Note: T here does not have to be repr(C) as we do the size calculation on the rust side
    #[repr(C)]
    struct TlsCell<T> {
        initialized: bool,
        value: MaybeUninit<T>,
    }

    pub struct TlsKey<T> {
        _marker: PhantomData<T>,
    }

    // this is kind of useless, but clippy wants this.
    impl<T: Default> Default for TlsKey<T> {
        fn default() -> Self {
            Self::new()
        }
    }

    impl<T> TlsKey<T> {
        pub const fn new() -> Self {
            Self {
                _marker: PhantomData,
            }
        }
    }

    impl<T: Default> TlsKey<T> {
        #[inline(always)]
        fn cell_ptr(&self, mt: &mut Metadata) -> *mut TlsCell<T> {
            // SAFETY: We assume dice correctly returns a pointer for TLS storage.
            // in debug build we do an additional sanity check that this holds.
            unsafe {
                let raw = raw::thread::self_tls(
                    mt,
                    self as *const _ as *const _,
                    size_of::<TlsCell<T>>(),
                );
                let ptr = raw as *mut TlsCell<T>;
                debug_assert!(ptr.is_aligned() && !ptr.is_null(), "Sanity Check");
                ptr
            }
        }

        #[inline]
        pub fn with<R>(&self, mt: &mut Metadata, f: impl FnOnce(&mut T) -> R) -> R {
            // SAFETY: see inside get_mut
            let t = unsafe { self.get_mut(mt) };
            f(t)
        }

        #[inline]
        /// # SAFETY
        /// this function requires a valid &'a mut Metdata, which forces unique and exclusive usage.
        /// Furthermore, as Metadata is !Send and !Sync, we also enfornce this thread local object will stay there.
        /// TODO: we could even consider Pin type.
        pub unsafe fn get_mut<'a>(&self, mt: &'a mut Metadata) -> &'a mut T {
            let cell = self.cell_ptr(mt);
            // TODO: consider using (#[cold] based) unlikely here as this only happens once
            // SAFETY: we know the ptr is not null and is correctly aligned from the cell_ptr() call
            if unsafe { !(*cell).initialized } {
                // SAFETY: external memory, so a volatile write
                unsafe { std::ptr::write_volatile((*cell).value.as_mut_ptr(), T::default()) };
                // SAFETY: this is safe from prior checks, set this to true so this won't be called multiple times.
                unsafe { (*cell).initialized = true };
            }
            // SAFETY: &'a mut Metadata ensures they have the same lifetime and metadata can only be borrowed once
            // as metadata is unique per subscribe call and is !Send & !Sync, this is safe.
            unsafe { &mut *(*cell).value.as_mut_ptr() }
        }
    }

    #[macro_export]
    macro_rules! tls_key {
        ($name:ident : $ty:ty) => {
            static $name: TlsKey<$ty> = TlsKey::new();
        };
    }
}

/// Create a callback and subscribe it to dice.
// this subscribe macro emulates a normal rust anonymous function structure.
// it creates a c callback and subscribes it automatically
// it is type, lifetime and capture guarded using the _guard
#[macro_export]
macro_rules! subscribe_scoped {
    ($chain:expr, $prio:expr, |$e:ident: &$t:ty, $m:ident| $body:block) => {{
        // this guard enforces no capturing of outside scope variables (except statics of course)
        // it also enforces lifetimes and types of the $body
        let _guard: fn(&$t, &mut $crate::Metadata) -> $crate::DiceResult =
            |$e: &$t, $m: &mut $crate::Metadata| $body;

        extern "C" fn __trampoline(
            chain: $crate::Chain,
            _ty: $crate::TypeId,
            event: *const core::ffi::c_void,
            md: *mut $crate::Metadata,
        ) -> $crate::DiceResult {
            // SAFETY: the dice subscribe callback either gives a correctly typed pointer for the event
            // or it gives a null in case the Event struct is empty.
            // in this case an empty Fallback is used.
            // Rust allows to take references of unit types directly and treat them as instances (like &() as () is type fields)
            // as these have no fields, there is also no concern of possibility of mutating these (potentially shared) references
            let Some(ev_ref) = (unsafe { <$t as $crate::DiceEvent>::from_raw(event as _) }) else {
                return $crate::DiceResult::Invalid;
            };

            let __chain = chain;

            let $e: &$t = ev_ref;
            let Some($m) = (unsafe { md.as_mut() }) else {
                return $crate::DiceResult::Invalid;
            };

            $body
        }

        // SAFETY
        unsafe {
            $crate::raw::ps_subscribe(
                $chain,
                <$t as $crate::DiceEvent>::ID,
                Some(__trampoline),
                $prio,
            )
        }
    }};
}

/// Create a callback and subscribe it to dice.
/// this is a convenience macro over `subscribe_scoped` which allows usage in global scope
/// and subscribes at startup of the program using a startup constructors.
/// `Once` ensures this only happens once, even in multithreaded applications.
#[macro_export]
macro_rules! subscribe {
    ($chain:expr, $slot:expr, |$e:ident: &$t:ty, $m:ident| $body:block) => {
        const _: () = {
            #[allow(non_snake_case)]
            #[::ctor::ctor]
            fn __dice_subscribe_ctor() {
                use ::std::sync::Once;
                static INIT: Once = Once::new();

                INIT.call_once(|| {
                    let _ = $crate::subscribe_scoped!($chain, $slot, |$e: &$t, $m| $body);
                });
            }
        };
    };
}

/// Helper to initialize logging
/// TODO: make it a struct
#[macro_export]
macro_rules! init_dice_state {
    (log_level: $level:expr) => {
        #[global_allocator]
        static GLOBAL: $crate::MempoolAllocator = $crate::MempoolAllocator;

        #[::ctor::ctor]
        fn __init_log() {
            $crate::log::init($level);
        }
    };
    () => { init_dice_state!(log_level: $crate::log::LogLevel::Debug); }
}
