# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is OCaml bindings to the Hugging Face Tokenizers library, using Rust FFI through the `ocaml-rs` crate. The project combines OCaml, Rust, and Dune build system to provide OCaml access to high-performance tokenizers.

## Development Commands

### Building the Project
```bash
dune build
```

### Running Tests
```bash
dune runtest
```

### Interactive Development
```bash
OCAML_INTEROP_NO_CAML_STARTUP=1 dune utop
```
The `OCAML_INTEROP_NO_CAML_STARTUP` environment variable is required for correct library linking.

### Building Individual Components
```bash
# Build only the library
dune build src/

# Build tests
dune build test/
```

## Architecture

### Hybrid Language Architecture
- **OCaml Interface**: `src/huggingface_tokenizers.ml` and `src/huggingface_tokenizers.mli` - auto-generated OCaml bindings
- **Rust Implementation**: `src/lib.rs` - actual tokenizer functionality using the `tokenizers` crate
- **Build Bridge**: `build.rs` - uses `ocaml-build` to generate OCaml signatures from Rust code

### Key Components
- **`src/lib.rs`**: Rust functions with `#[ocaml::func]` and `#[ocaml::sig]` attributes for FFI
- **`src/huggingface_tokenizers.ml*`**: Generated OCaml bindings (do not edit manually)
- **`test/test.ml`**: Integration tests demonstrating usage

### Build Process
1. `cargo build --release` compiles Rust code to static/dynamic libraries
2. `dune` build system links the libraries with OCaml code
3. `build.rs` generates OCaml signatures from Rust function signatures

## Dependencies

### OCaml Dependencies
- `ocaml >= 4.03.0`
- `dune >= 1.5`
- `conf-rust-2024` (ensures Rust toolchain availability)

### Rust Dependencies
- `ocaml-rs` crate for OCaml FFI
- `tokenizers` crate for Hugging Face tokenizers functionality

## Important Notes

- The project uses vendored Rust dependencies in the `vendor/` directory
- OCaml bindings are auto-generated - modify `src/lib.rs` and rebuild to update them
- The `target/` directory contains Rust build artifacts
- Library artifacts are generated as `libhuggingface_tokenizers.a` and `dllhuggingface_tokenizers.so` (copied from Rust's `libocaml_huggingface_tokenizers.*`)