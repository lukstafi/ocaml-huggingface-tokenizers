(* Hugging Face Tokenizers OCaml bindings *)

(* Tokenizer type - represented as int64 ID internally *)
type tokenizer = int64

(* Loading functions *)
external load_from_file: string -> (tokenizer, string) result = "load_from_file"

(* Vocabulary operations *)
external get_vocab_size: tokenizer -> int option = "get_vocab_size"
external token_to_id: tokenizer -> string -> int32 option = "token_to_id"
external id_to_token: tokenizer -> int32 -> string option = "id_to_token"

(* Core tokenization functions *)
external encode_text: tokenizer -> string -> bool -> (int32 array, string) result = "encode_text"
external decode_tokens: tokenizer -> int32 array -> bool -> (string, string) result = "decode_tokens"
external encode_with_attention: tokenizer -> string -> bool -> ((int32 array * int array), string) result = "encode_with_attention"
external encode_batch: tokenizer -> string array -> bool -> (int32 array array, string) result = "encode_batch"

(* Resource management *)
external free_tokenizer: tokenizer -> unit = "free_tokenizer"

(* Test function *)
external hello_world: unit -> string = "hello_world"

(* Convenience functions *)
val result_to_option: ('a, 'b) result -> 'a option

val encode: tokenizer -> string -> ?add_special_tokens:bool -> unit -> (int32 array, string) result
val decode: tokenizer -> int32 array -> ?skip_special_tokens:bool -> unit -> (string, string) result
val encode_with_mask: tokenizer -> string -> ?add_special_tokens:bool -> unit -> ((int32 array * int array), string) result

(* Helper functions for working with results *)
val unwrap_result_exn: ('a, string) result -> 'a
val map_error: ('a -> 'b) -> ('c, 'a) result -> ('c, 'b) result
