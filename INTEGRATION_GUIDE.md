# OCaml Hugging Face Tokenizers Integration Guide

## Overview

This project provides OCaml bindings for the Rust `tokenizers` library, enabling deep learning frameworks to use state-of-the-art tokenizers for models like GPT-2 and LLaMA.

## Current Implementation Status

### ✅ Completed Features

1. **Basic Tokenizer Loading**
   - Load tokenizer from JSON file: `load_from_file: string -> (tokenizer, string) result`
   - Resource management with `free_tokenizer: tokenizer -> unit`

2. **Vocabulary Operations**
   - Get vocabulary size: `get_vocab_size: tokenizer -> int option`
   - Token to ID conversion: `token_to_id: tokenizer -> string -> int32 option`
   - ID to token conversion: `id_to_token: tokenizer -> int32 -> string option`

3. **Core Tokenization Functions**
   - Text encoding: `encode_text: tokenizer -> string -> bool -> (int32 array, string) result`
   - Token decoding: `decode_tokens: tokenizer -> int32 array -> bool -> (string, string) result`
   - Encoding with attention masks: `encode_with_attention: tokenizer -> string -> bool -> ((int32 array * int array), string) result`
   - Batch encoding: `encode_batch: tokenizer -> string array -> bool -> (int32 array array, string) result`

4. **Convenience Functions**
   - Optional parameter wrappers: `encode`, `decode`, `encode_with_mask`
   - Result handling utilities: `result_to_option`, `unwrap_result_exn`, `map_error`

### Architecture

- **Tokenizer Storage**: Uses a global HashMap with int64 IDs for safe tokenizer management
- **Memory Safety**: Proper resource cleanup with `free_tokenizer`
- **Error Handling**: Comprehensive Result types for all operations
- **Type Safety**: Strong typing with custom `tokenizer` type (int64 internally)

## Deep Learning Framework Integration

### For GPT-2 Support

```ocaml
(* Load GPT-2 tokenizer *)
let gpt2_tokenizer = match load_from_file "gpt2_tokenizer.json" with
  | Ok tok -> tok
  | Error msg -> failwith ("Failed to load GPT-2 tokenizer: " ^ msg)

(* Encode text for training/inference *)
let encode_for_training text =
  match encode_text gpt2_tokenizer text true with
  | Ok tokens -> tokens
  | Error msg -> failwith ("Encoding failed: " ^ msg)

(* Get attention mask for padding *)
let encode_with_padding text =
  match encode_with_attention gpt2_tokenizer text true with
  | Ok (tokens, attention_mask) -> (tokens, attention_mask)
  | Error msg -> failwith ("Encoding with attention failed: " ^ msg)
```

### For LLaMA Support

```ocaml
(* Load LLaMA tokenizer (SentencePiece-based) *)
let llama_tokenizer = match load_from_file "llama_tokenizer.json" with
  | Ok tok -> tok
  | Error msg -> failwith ("Failed to load LLaMA tokenizer: " ^ msg)

(* Batch processing for efficient inference *)
let encode_batch_for_inference texts =
  match encode_batch llama_tokenizer texts true with
  | Ok token_arrays -> token_arrays
  | Error msg -> failwith ("Batch encoding failed: " ^ msg)
```

## Required Tokenizer Files

### GPT-2
- **File**: `gpt2_tokenizer.json`
- **Source**: Can be downloaded from Hugging Face Hub
- **Features**: BPE tokenization, vocabulary size ~50,257

### LLaMA
- **File**: `llama_tokenizer.json` 
- **Source**: Can be downloaded from Hugging Face Hub (llama-2-7b-hf, etc.)
- **Features**: SentencePiece tokenization, vocabulary size ~32,000

## Integration Requirements for DL Frameworks

### 1. Tokenizer Loading
```ocaml
(* Framework should provide a tokenizer loading function *)
val load_model_tokenizer : string -> tokenizer
```

### 2. Preprocessing Pipeline
```ocaml
(* Convert text to model inputs *)
val preprocess_text : tokenizer -> string -> int32 array * int array
```

### 3. Postprocessing Pipeline
```ocaml
(* Convert model outputs back to text *)
val postprocess_tokens : tokenizer -> int32 array -> string
```

### 4. Batch Processing
```ocaml
(* Efficient batch processing for training *)
val preprocess_batch : tokenizer -> string array -> (int32 array array * int array array)
```

## Usage Examples

### Simple Text Generation
```ocaml
open Huggingface_tokenizers

let () =
  (* Load tokenizer *)
  let tokenizer = match load_from_file "gpt2_tokenizer.json" with
    | Ok tok -> tok
    | Error msg -> failwith msg
  in
  
  (* Encode prompt *)
  let prompt = "The quick brown fox" in
  let tokens = match encode_text tokenizer prompt true with
    | Ok tokens -> tokens
    | Error msg -> failwith msg
  in
  
  (* ... pass tokens to model for inference ... *)
  
  (* Decode generated tokens *)
  let generated_tokens = [|464; 2068; 7586; 21831|] in (* example *)
  let decoded = match decode_tokens tokenizer generated_tokens true with
    | Ok text -> text
    | Error msg -> failwith msg
  in
  
  Printf.printf "Generated: %s\n" decoded;
  
  (* Clean up *)
  free_tokenizer tokenizer
```

### Training Data Preparation
```ocaml
let prepare_training_data texts =
  let tokenizer = match load_from_file "model_tokenizer.json" with
    | Ok tok -> tok
    | Error msg -> failwith msg
  in
  
  let token_arrays = match encode_batch tokenizer texts true with
    | Ok arrays -> arrays
    | Error msg -> failwith msg
  in
  
  (* Convert to training format *)
  let training_data = Array.map (fun tokens ->
    let input_ids = Array.sub tokens 0 (Array.length tokens - 1) in
    let labels = Array.sub tokens 1 (Array.length tokens - 1) in
    (input_ids, labels)
  ) token_arrays in
  
  free_tokenizer tokenizer;
  training_data
```

## Testing and Validation

The current implementation has been tested with:
- ✅ Basic function calls and memory management
- ✅ Rust-OCaml binding compilation
- ✅ Error handling and result types

**Next Steps for Production Use:**
1. Download actual tokenizer files (GPT-2, LLaMA)
2. Test with real tokenizer configurations
3. Benchmark performance with large texts
4. Add more specialized functions as needed by specific frameworks

## Performance Considerations

- **Memory Management**: Tokenizers are stored globally and managed by ID
- **Batch Processing**: Efficient batch encoding for training workloads
- **Error Handling**: Comprehensive error reporting without crashes
- **Resource Cleanup**: Explicit tokenizer cleanup to prevent memory leaks

This implementation provides a solid foundation for integrating Hugging Face tokenizers into OCaml-based deep learning frameworks, with specific support for the tokenization patterns used by GPT-2 and LLaMA models.
