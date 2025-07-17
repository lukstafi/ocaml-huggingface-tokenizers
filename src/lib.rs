use tokenizers::Tokenizer;
use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::LazyLock;

// Global storage for tokenizers using a simple ID system
static TOKENIZERS: LazyLock<Mutex<HashMap<u64, Tokenizer>>> = LazyLock::new(|| Mutex::new(HashMap::new()));
static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

// Load tokenizer from file and return an ID
#[ocaml::func]
#[ocaml::sig("string -> (int64, string) result")]
pub fn load_from_file(path: String) -> Result<i64, String> {
    let tokenizer = Tokenizer::from_file(path)
        .map_err(|e| e.to_string())?;
    
    let id = NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    
    {
        let mut store = TOKENIZERS.lock().unwrap();
        store.insert(id, tokenizer);
    }
    
    Ok(id as i64)
}

// Get vocabulary size
#[ocaml::func]
#[ocaml::sig("int64 -> int option")]
pub fn get_vocab_size(id: i64) -> Option<usize> {
    let store = TOKENIZERS.lock().unwrap();
    store.get(&(id as u64)).map(|t| t.get_vocab_size(true))
}

// Convert token to ID
#[ocaml::func]
#[ocaml::sig("int64 -> string -> int32 option")]
pub fn token_to_id(id: i64, token: String) -> Option<i32> {
    let store = TOKENIZERS.lock().unwrap();
    store.get(&(id as u64))
        .and_then(|t| t.token_to_id(&token))
        .map(|id| id as i32)
}

// Convert ID to token
#[ocaml::func]
#[ocaml::sig("int64 -> int32 -> string option")]
pub fn id_to_token(id: i64, token_id: i32) -> Option<String> {
    let store = TOKENIZERS.lock().unwrap();
    store.get(&(id as u64))
        .and_then(|t| t.id_to_token(token_id as u32))
}

// Encode text to token IDs
#[ocaml::func]
#[ocaml::sig("int64 -> string -> bool -> (int32 array, string) result")]
pub fn encode_text(id: i64, text: String, add_special_tokens: bool) -> Result<Vec<i32>, String> {
    let store = TOKENIZERS.lock().unwrap();
    match store.get(&(id as u64)) {
        Some(tokenizer) => {
            let encoding = tokenizer
                .encode(text, add_special_tokens)
                .map_err(|e| e.to_string())?;
            
            let ids = encoding.get_ids();
            let ids_i32: Vec<i32> = ids.iter().map(|&id| id as i32).collect();
            Ok(ids_i32)
        }
        None => Err("Invalid tokenizer ID".to_string())
    }
}

// Decode token IDs to text
#[ocaml::func]
#[ocaml::sig("int64 -> int32 array -> bool -> (string, string) result")]
pub fn decode_tokens(id: i64, ids: Vec<i32>, skip_special_tokens: bool) -> Result<String, String> {
    let store = TOKENIZERS.lock().unwrap();
    match store.get(&(id as u64)) {
        Some(tokenizer) => {
            let ids_u32: Vec<u32> = ids.iter().map(|&id| id as u32).collect();
            let text = tokenizer
                .decode(&ids_u32, skip_special_tokens)
                .map_err(|e| e.to_string())?;
            Ok(text)
        }
        None => Err("Invalid tokenizer ID".to_string())
    }
}

// Encode with attention mask
#[ocaml::func]
#[ocaml::sig("int64 -> string -> bool -> ((int32 array * int array), string) result")]
pub fn encode_with_attention(id: i64, text: String, add_special_tokens: bool) -> Result<(Vec<i32>, Vec<i32>), String> {
    let store = TOKENIZERS.lock().unwrap();
    match store.get(&(id as u64)) {
        Some(tokenizer) => {
            let encoding = tokenizer
                .encode(text, add_special_tokens)
                .map_err(|e| e.to_string())?;
            
            let ids = encoding.get_ids();
            let ids_i32: Vec<i32> = ids.iter().map(|&id| id as i32).collect();
            
            let attention_mask = encoding.get_attention_mask();
            let attention_i32: Vec<i32> = attention_mask.iter().map(|&mask| mask as i32).collect();
            
            Ok((ids_i32, attention_i32))
        }
        None => Err("Invalid tokenizer ID".to_string())
    }
}

// Batch encode
#[ocaml::func]
#[ocaml::sig("int64 -> string array -> bool -> (int32 array array, string) result")]
pub fn encode_batch(id: i64, texts: Vec<String>, add_special_tokens: bool) -> Result<Vec<Vec<i32>>, String> {
    let store = TOKENIZERS.lock().unwrap();
    match store.get(&(id as u64)) {
        Some(tokenizer) => {
            let encodings = tokenizer
                .encode_batch(texts, add_special_tokens)
                .map_err(|e| e.to_string())?;
            
            let mut result = Vec::new();
            for encoding in encodings {
                let ids = encoding.get_ids();
                let ids_i32: Vec<i32> = ids.iter().map(|&id| id as i32).collect();
                result.push(ids_i32);
            }
            Ok(result)
        }
        None => Err("Invalid tokenizer ID".to_string())
    }
}

// Free tokenizer resources
#[ocaml::func]
#[ocaml::sig("int64 -> unit")]
pub fn free_tokenizer(id: i64) {
    let mut store = TOKENIZERS.lock().unwrap();
    store.remove(&(id as u64));
}

// Test function to verify the binding works
#[ocaml::func]
#[ocaml::sig("unit -> string")]
pub fn hello_world() -> &'static str {
    "hello, world!"
}
