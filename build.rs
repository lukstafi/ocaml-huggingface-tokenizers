pub fn main() -> std::io::Result<()> {
    ocaml_build::Sigs::new("src/huggingface_tokenizers.ml").generate()
}
