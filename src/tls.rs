use once_cell::sync::Lazy;

static RUSTLS_CRYPTO_PROVIDER_INIT: Lazy<()> = Lazy::new(|| {
    let _ = rustls::crypto::ring::default_provider().install_default();
});

pub fn ensure_rustls_crypto_provider() {
    Lazy::force(&RUSTLS_CRYPTO_PROVIDER_INIT);
}
