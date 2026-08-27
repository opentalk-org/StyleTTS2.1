use std::path::PathBuf;

use anyhow::Context;
use aws_config::BehaviorVersion;
use aws_sdk_s3::config::Credentials;
use clap::{
    Parser,
    builder::{
        Styles,
        styling::{AnsiColor, Effects},
    },
};
use sqlx::PgPool;
use tokio::fs;

use crate::server::serve;

mod audio;
mod db;
mod loader;
mod prefetch;
mod sampling;
mod server;
mod session;
mod symbols;

const STYLES: Styles = Styles::styled()
    .header(AnsiColor::Green.on_default().effects(Effects::BOLD))
    .usage(AnsiColor::Green.on_default().effects(Effects::BOLD))
    .literal(AnsiColor::Cyan.on_default().effects(Effects::BOLD))
    .placeholder(AnsiColor::Cyan.on_default())
    .error(AnsiColor::Red.on_default().effects(Effects::BOLD))
    .valid(AnsiColor::Cyan.on_default().effects(Effects::BOLD))
    .invalid(AnsiColor::Yellow.on_default().effects(Effects::BOLD));

#[derive(Parser)]
#[command(name = "givemedata a", styles = STYLES)]
#[command(about = "GIVE ME DATA!", long_about = None)]
struct Args {
    #[arg(
        short,
        long,
        env = "PORT",
        help = "Main server binding port.",
        default_value = "8181"
    )]
    port: u16,
    #[arg(
        long,
        env = "DATABASE_URL",
        help = "PostgreSQL URI to database with data metadata."
    )]
    db_url: String,
    #[arg(long, env = "AWS_ENDPOINT_URL", help = "Endpoint to S3 bucket.")]
    s3_endpoint: String,
    #[arg(long, env = "AWS_ACCESS_KEY_ID", help = "S3 access key ID.")]
    s3_key: String,
    #[arg(long, env = "AWS_SECRET_ACCESS_KEY", help = "S3 secret key.")]
    s3_secret: String,
    #[arg(long, env = "S3_BUCKET", help = "S3 bucket name.")]
    bucket: String,
    #[arg(
        long,
        env = "CACHE_DIR",
        help = "Directory for spilling prefetched audio to disk; omit to keep it in memory."
    )]
    cache_dir: PathBuf,
    #[arg(
        long,
        env = "SYNTHETIC",
        help = "Serve synthetic sessions: fabricated samples instead of database rows and bucket audio."
    )]
    synthetic: bool,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("debug,h2=off,hyper=off,tower=off,sqlx=off,aws_runtime=off,aws_sdk_s3=off")
        .init();

    let args = Args::parse();

    let pg_pool = PgPool::connect(&args.db_url)
        .await
        .context(format!("failed to connect to postgres"))?;

    let s3_config = aws_config::defaults(BehaviorVersion::latest())
        .endpoint_url(&args.s3_endpoint)
        .credentials_provider(Credentials::new(
            &args.s3_key,
            &args.s3_secret,
            None,
            None,
            "r2",
        ))
        .load()
        .await;
    let s3_client = aws_sdk_s3::Client::from_conf(
        aws_sdk_s3::config::Builder::from(&s3_config)
            .force_path_style(true)
            .build(),
    );

    fs::create_dir_all(&args.cache_dir).await?;

    serve(
        args.port,
        s3_client,
        pg_pool,
        args.bucket.leak(),
        Box::leak(args.cache_dir.into_boxed_path()),
        args.synthetic,
    )
    .await
}
