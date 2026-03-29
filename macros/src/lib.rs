use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput, Data, Fields};

#[proc_macro_derive(Module)]
pub fn derive_module(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;

    let fields = match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(fields) => &fields.named,
            _ => panic!("Module can only be derived for structs with named fields"),
        },
        _ => panic!("Module can only be derived for structs"),
    };

    let recurse = fields.iter().map(|f| {
        let name = &f.ident;
        quote! {
            params.extend(self.#name.parameters());
        }
    });

    let expanded = quote! {
        impl nexus_ml::nn::Module for #name {
            fn parameters(&self) -> Vec<nexus_ml::Tensor> {
                let mut params = Vec::new();
                #(#recurse)*
                params
            }
        }
    };

    TokenStream::from(expanded)
}
