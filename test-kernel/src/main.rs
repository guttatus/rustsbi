// A test kernel to test RustSBI function on all platforms
#![feature(naked_functions, global_asm, asm, llvm_asm)]
#![no_std]
#![no_main]

#[macro_use]
mod console;
mod sbi;

use riscv::register::{sepc, stvec::{self, TrapMode}};
use core::panic::PanicInfo;

#[cfg_attr(not(test), panic_handler)]
#[allow(unused)]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}

const BOOT_STACK_SIZE: usize = 4096 * 4 * 8;

static mut BOOT_STACK: [u8; BOOT_STACK_SIZE] = [0; BOOT_STACK_SIZE];

#[naked]
#[link_section = ".text.entry"] 
#[export_name = "_start"]
unsafe extern "C" fn entry() -> ! {
    asm!("
    # 1. set sp
    # sp = bootstack + (hartid + 1) * 0x10000
    add     t0, a0, 1
    slli    t0, t0, 14
1:  auipc   sp, %pcrel_hi({boot_stack})
    addi    sp, sp, %pcrel_lo(1b)
    add     sp, sp, t0

    # 2. jump to rust_main (absolute address)
1:  auipc   t0, %pcrel_hi({rust_main})
    addi    t0, t0, %pcrel_lo(1b)
    jr      t0
    ", 
    boot_stack = sym BOOT_STACK, 
    rust_main = sym rust_main,
    options(noreturn))
}

pub extern "C" fn rust_main(hartid: usize, dtb_pa: usize) -> ! {
    println!("<< Test-kernel: Hart id = {}, DTB physical address = {:#x}", hartid, dtb_pa);
    unsafe { stvec::write(start_trap as usize, TrapMode::Direct) };
    println!(">> Test-kernel: Trigger illegal exception");
    unsafe { asm!("unimp") };
    println!("<< Test-kernel: SBI test success, shutdown");
    sbi::shutdown()
}

#[naked]
#[link_section = ".text"]
unsafe extern "C" fn start_trap() {
    asm!("
.altmacro
.macro STORE reg, offset
    sd  \\reg, \\offset* {REGBYTES} (sp)
.endm
.macro LOAD reg, offset
    ld  \\reg, \\offset* {REGBYTES} (sp)
.endm
    addi    sp, sp, -16 * {REGBYTES}
    STORE   ra, 0
    STORE   t0, 1
    STORE   t1, 2
    STORE   t2, 3
    STORE   t3, 4
    STORE   t4, 5
    STORE   t5, 6
    STORE   t6, 7
    STORE   a0, 8
    STORE   a1, 9
    STORE   a2, 10
    STORE   a3, 11
    STORE   a4, 12
    STORE   a5, 13
    STORE   a6, 14
    STORE   a7, 15
    mv      a0, sp
    call    {rust_trap_exception}
    LOAD    ra, 0
    LOAD    t0, 1
    LOAD    t1, 2
    LOAD    t2, 3
    LOAD    t3, 4
    LOAD    t4, 5
    LOAD    t5, 6
    LOAD    t6, 7
    LOAD    a0, 8
    LOAD    a1, 9
    LOAD    a2, 10
    LOAD    a3, 11
    LOAD    a4, 12
    LOAD    a5, 13
    LOAD    a6, 14
    LOAD    a7, 15
    addi    sp, sp, 16 * {REGBYTES}
    sret
    ",
    REGBYTES = const core::mem::size_of::<usize>(),
    rust_trap_exception = sym rust_trap_exception,
    options(noreturn))
}

pub extern "C" fn rust_trap_exception() {
    println!("<< Test-kernel: Illegal exception");
    sepc::write(sepc::read().wrapping_add(4));
}
