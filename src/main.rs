use std::path::PathBuf;
use structopt::StructOpt;
use image::{GenericImageView, open};

mod error;
use crate::error::Result;

mod op;
use op::run_pipeline;

mod stack;

#[derive(StructOpt, Debug)]
struct Opt {
    /// Verbose mode (-v, -vv, -vvv, etc.)
    #[structopt(short, long, parse(from_occurrences))]
    verbose: u8,

    /// Input file
    #[structopt(short, long, parse(from_os_str))]
    input: PathBuf,

    /// Output file
    #[structopt(short, long, parse(from_os_str))]
    output: PathBuf,

    /// Image processing pipeline to apply
    #[structopt(short, long)]
    pipeline: Option<String>,
}

fn main() -> Result<()> {
    let opt = Opt::from_args();
    let verbose = opt.verbose > 0;
    let input = open(&opt.input)?;
    if verbose {
        println!("Input image {:?} (width: {}, height: {})", &opt.input, input.width(), input.height());
    }
    let output = if opt.pipeline.is_some() {
        run_pipeline(&opt.pipeline.unwrap(), vec![input], opt.verbose > 0)
    } else {
        input
    };
    if verbose {
        println!("Output image {:?} (width: {}, height: {})", &opt.output, output.width(), output.height());
    }
    output.save(&opt.output)?;

    Ok(())
}
