use imagecli::{documentation::generate_guide, process};
use structopt::StructOpt;

#[derive(StructOpt, Debug)]
struct Opt {
    /// Enable verbose logging.
    #[structopt(short, long)]
    verbose: bool,

    /// Input files or glob patterns.
    #[structopt(short, long)]
    input: Vec<String>,

    /// Output files.
    #[structopt(short, long)]
    output: Vec<String>,

    /// Image processing pipeline to apply.
    #[structopt(short, long)]
    pipeline: Option<String>,

    /// Ignore all the other flags and regenerate a user guide instead.
    #[structopt(long)]
    generate_guide: bool,

    /// Enable parallelisation if possible.
    #[structopt(long)]
    parallel: bool,
}

fn main() {
    let opt = Opt::from_args();

    let result = if opt.generate_guide {
        generate_guide(opt.verbose)
    } else {
        process(
            &opt.input,
            &opt.output,
            opt.pipeline,
            opt.verbose,
            opt.parallel,
        )
    };

    match result {
        Ok(_) => {}
        Err(e) => panic!("{}", e),
    }
}
