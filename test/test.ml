open Huggingface_tokenizers

let test_hello_world () =
  let result = hello_world () in
  Printf.printf "Hello world test: %s\n" result

let test_basic_encoding () =
  (* This test will fail initially as we need a tokenizer file, but shows the API *)
  match load_from_file "test_tokenizer.json" with
  | Ok tokenizer ->
    Printf.printf "Tokenizer loaded successfully with ID: %Ld\n" tokenizer;
    
    (match get_vocab_size tokenizer with
     | Some size -> Printf.printf "Vocab size: %d\n" size
     | None -> Printf.printf "Failed to get vocab size\n");
    
    (* Test basic encoding *)
    (match encode_text tokenizer "Hello world" true with
     | Ok ids ->
       Printf.printf "Encoded successfully, length: %d\n" (Array.length ids);
       if Array.length ids > 0 then
         Printf.printf "First token ID: %ld\n" ids.(0)
     | Error msg ->
       Printf.printf "Encoding failed: %s\n" msg);
    
    (* Test token-to-id conversion *)
    (match token_to_id tokenizer "hello" with
     | Some id -> Printf.printf "Token 'hello' has ID: %ld\n" id
     | None -> Printf.printf "Token 'hello' not found\n");
    
    (* Test decoding *)
    (match encode_text tokenizer "Hello world" true with
     | Ok ids ->
       (match decode_tokens tokenizer ids true with
        | Ok decoded -> Printf.printf "Decoded text: %s\n" decoded
        | Error msg -> Printf.printf "Decoding failed: %s\n" msg)
     | Error msg ->
       Printf.printf "Encoding for decode test failed: %s\n" msg);
    
    (* Test with attention mask *)
    (match encode_with_attention tokenizer "Hello world" true with
     | Ok (ids, attention) ->
       Printf.printf "Encoded with attention: %d tokens, %d attention values\n" 
         (Array.length ids) (Array.length attention)
     | Error msg ->
       Printf.printf "Encoding with attention failed: %s\n" msg);
    
    (* Clean up *)
    free_tokenizer tokenizer;
    Printf.printf "Tokenizer freed\n"
    
  | Error msg ->
    Printf.printf "Failed to load tokenizer: %s\n" msg

let () =
  test_hello_world ();
  test_basic_encoding ();
  Printf.printf "Tests completed\n"
