(* Example showing how a deep learning framework would integrate with the tokenizers *)

open Huggingface_tokenizers

(* Example framework module *)
module DeepLearningFramework = struct
  
  (* Framework-specific tokenizer wrapper *)
  type model_tokenizer = {
    tokenizer_id: tokenizer;
    vocab_size: int;
    model_name: string;
  }
  
  (* Load model tokenizer with framework-specific setup *)
  let load_model_tokenizer model_name tokenizer_path =
    match load_from_file tokenizer_path with
    | Ok tokenizer_id ->
        (match get_vocab_size tokenizer_id with
         | Some vocab_size -> 
             Ok { tokenizer_id; vocab_size; model_name }
         | None -> 
             free_tokenizer tokenizer_id;
             Error "Failed to get vocabulary size")
    | Error msg -> Error ("Failed to load tokenizer: " ^ msg)
  
  (* Preprocess text for model input *)
  let preprocess_text model_tok text =
    match encode_with_attention model_tok.tokenizer_id text true with
    | Ok (input_ids, attention_mask) -> Ok (input_ids, attention_mask)
    | Error msg -> Error ("Preprocessing failed: " ^ msg)
  
  (* Preprocess batch for training *)
  let preprocess_batch model_tok texts =
    match encode_batch model_tok.tokenizer_id texts true with
    | Ok token_arrays -> 
        (* Generate attention masks for each sequence *)
        let attention_masks = Array.map (fun tokens ->
          Array.make (Array.length tokens) 1 (* all tokens are real, no padding *)
        ) token_arrays in
        Ok (token_arrays, attention_masks)
    | Error msg -> Error ("Batch preprocessing failed: " ^ msg)
  
  (* Generate text from model outputs *)
  let generate_text model_tok token_ids =
    match decode_tokens model_tok.tokenizer_id token_ids true with
    | Ok text -> Ok text
    | Error msg -> Error ("Text generation failed: " ^ msg)
  
  (* Clean up model tokenizer *)
  let free_model_tokenizer model_tok =
    free_tokenizer model_tok.tokenizer_id
  
  (* Example training data preparation *)
  let prepare_training_data model_tok text_samples =
    let process_sample text =
      match preprocess_text model_tok text with
      | Ok (input_ids, attention_mask) ->
          (* Create labels by shifting input_ids by 1 position *)
          let labels = Array.sub input_ids 1 (Array.length input_ids - 1) in
          let input_ids = Array.sub input_ids 0 (Array.length input_ids - 1) in
          let attention_mask = Array.sub attention_mask 0 (Array.length attention_mask - 1) in
          Some (input_ids, attention_mask, labels)
      | Error _ -> None
    in
    Array.to_list text_samples
    |> List.filter_map process_sample
    |> Array.of_list
end

(* Example usage for GPT-2 *)
let demo_gpt2 () =
  print_endline "=== GPT-2 Integration Demo ===";
  
  (* This would work with a real GPT-2 tokenizer file *)
  match DeepLearningFramework.load_model_tokenizer "gpt2" "gpt2_tokenizer.json" with
  | Ok model_tokenizer ->
      Printf.printf "Loaded GPT-2 tokenizer with vocab size: %d\n" model_tokenizer.vocab_size;
      
      (* Example preprocessing *)
      let sample_text = "The quick brown fox jumps over the lazy dog" in
      (match DeepLearningFramework.preprocess_text model_tokenizer sample_text with
       | Ok (input_ids, attention_mask) ->
           Printf.printf "Preprocessed text: %d tokens\n" (Array.length input_ids);
           Printf.printf "Sample token IDs: [%s]\n" 
             (String.concat "; " (Array.to_list input_ids |> List.map Int32.to_string));
       | Error msg -> Printf.printf "Preprocessing error: %s\n" msg);
      
      (* Example batch processing *)
      let batch_texts = [|
        "Hello world"; 
        "This is a test"; 
        "Deep learning with OCaml"
      |] in
      (match DeepLearningFramework.preprocess_batch model_tokenizer batch_texts with
       | Ok (token_arrays, attention_arrays) ->
           Printf.printf "Batch processed: %d samples\n" (Array.length token_arrays);
           Array.iteri (fun i tokens ->
             Printf.printf "  Sample %d: %d tokens\n" i (Array.length tokens)
           ) token_arrays;
       | Error msg -> Printf.printf "Batch processing error: %s\n" msg);
      
      DeepLearningFramework.free_model_tokenizer model_tokenizer;
      print_endline "GPT-2 tokenizer demo completed"
      
  | Error msg -> Printf.printf "Failed to load GPT-2 tokenizer: %s\n" msg

(* Example usage for LLaMA *)
let demo_llama () =
  print_endline "\n=== LLaMA Integration Demo ===";
  
  (* This would work with a real LLaMA tokenizer file *)
  match DeepLearningFramework.load_model_tokenizer "llama" "llama_tokenizer.json" with
  | Ok model_tokenizer ->
      Printf.printf "Loaded LLaMA tokenizer with vocab size: %d\n" model_tokenizer.vocab_size;
      
      (* Example training data preparation *)
      let training_texts = [|
        "The capital of France is Paris.";
        "Machine learning is a subset of artificial intelligence.";
        "OCaml is a functional programming language."
      |] in
      
      let training_data = DeepLearningFramework.prepare_training_data model_tokenizer training_texts in
      Printf.printf "Prepared %d training samples\n" (Array.length training_data);
      
      DeepLearningFramework.free_model_tokenizer model_tokenizer;
      print_endline "LLaMA tokenizer demo completed"
      
  | Error msg -> Printf.printf "Failed to load LLaMA tokenizer: %s\n" msg

(* Basic demonstration with hello world *)
let demo_basic () =
  print_endline "=== Basic Tokenizer Demo ===";
  let result = hello_world () in
  Printf.printf "Hello world test: %s\n" result;
  print_endline "Basic demo completed"

(* Main demo *)
let () =
  demo_basic ();
  demo_gpt2 ();
  demo_llama ();
  print_endline "\nAll demos completed!"
