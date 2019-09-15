//! Functions for handling the very basic templating used in README_template.txt
//!
//! The following forms are supported.
//!
//! ```text
//! $TABLE_OF_CONTENTS
//! ```
//! This is replaced by a table of contents, containing links to
//! all headers in the generated document of level >= 2 and <= 3.
//!
//! ```text
//! $OPERATIONS
//! ```
//! This is replaced by a table listing summary information for all
//! of the image operations, and a detailed listing of the operations
//! and their example usages.
//!
//! ```text
//! $EXAMPLE(
//!     key1: value1
//!     key2: value2
//! )$
//! ```
//! This defines an example pipeline to run. These examples are run while
//! generating the documentation, and their results included in the docs, along
//! with the command lines required to generate them. See parse_example for the
//! options that are supported.
//!
//! ```text
//! $STACK_DIAGRAM(
//!     some text
//!     STACK A B C
//!     some other text
//! )$
//! ```
//! This defines a diagram to render in the readme to demonstrate the state of
//! the image stack during the evaluation of a pipeline. Lines starting STACK
//! indicate that a stack should be rendered with the given elements. Other lines
//! are used to label transitions between stacks.
//!

use crate::example::{Example, Style};
use super::stack_diagram::{StackDiagram, StackDiagramStage};

/// Parses a `$EXAMPLE(...)` from a template.
/// Currently this is a bit clunky - it requires the opening line `$EXAMPLE(`
/// to have already been consumed.
pub fn parse_example<'a, I: Iterator<Item=&'a str>>(lines: &mut I) -> Example {
    let mut example = Example::new(1, 1, "");
    for line in lines {
        if line.ends_with(")$") {
            break;
        }
        let split: Vec<_> = line.trim().split(':').collect();
        if split.len() != 2 {
            continue;
        }
        let name = split[0].trim();
        let value = split[1].trim();
        match name {
            "pipeline" => example.pipeline = value.into(),
            "num_inputs" => example.num_inputs = value.parse::<usize>().unwrap(),
            "num_outputs" => example.num_outputs = value.parse::<usize>().unwrap(),
            "inputs" => {
                example.override_input_names =
                    Some(value.split_whitespace().map(|s| s.to_string()).collect())
            }
            "output_file_extension" => example.output_file_extension = Some(value.into()),
            "style" => {
                example.style = match value {
                    "longform" => Style::LongForm,
                    "shortform" => Style::ShortForm,
                    _ => panic!("Invalid style"),
                }
            }
            _ => {}
        }
    }
    example
}

/// Parses a `$STACK_DIAGRAM(...)` from a template.
/// Currently this is a bit clunky - it requires the opening line `$STACK_DIAGRAM(`
/// to have already been consumed.
pub fn parse_diagram<'a, I: Iterator<Item=&'a str>>(lines: &mut I) -> StackDiagram {
    let mut stages = Vec::new();
    for line in lines {
        if line.ends_with(")$") {
            break;
        }
        let stage = if line.trim().starts_with("STACK") {
            let line = line.replace("STACK", "");
            let line = line.trim();
            let elements = line.split_whitespace().map(|s| s.to_string()).collect();
            StackDiagramStage::Stack(elements)
        } else {
            StackDiagramStage::Transition(line.trim().to_string())
        };
        stages.push(stage);
    }
    StackDiagram { stages }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::example::Style;

    #[test]
    fn test_parse_diagram() {
        // The opening $STACK_DIAGRAM is handled in render_readme.
        // This is a bit clunky, but it works for now.
        let template = "    some words
    STACK A B C
    more words
)$";
        assert_eq!(
            parse_diagram(&mut template.lines()),
            StackDiagram {
                stages: vec![
                    StackDiagramStage::Transition("some words".into()),
                    StackDiagramStage::Stack(vec!["A".into(), "B".into(), "C".into()]),
                    StackDiagramStage::Transition("more words".into()),
                ]
            }
        );
    }

    #[test]
    fn test_parse_example() {
        // The opening $EXAMPLE is handled in render_readme.
        // This is a bit clunky, but it works for now.
        let template = "    num_inputs: 3
    num_outputs: 2
    pipeline: gray > rotate 45
    inputs: foo bar baz
    output_file_extension: jpg
    style: longform
)$";
        assert_eq!(
            parse_example(&mut template.lines()),
            Example {
                num_inputs: 3,
                num_outputs: 2,
                pipeline: "gray > rotate 45".into(),
                override_input_names: Some(vec!["foo".into(), "bar".into(), "baz".into()]),
                output_file_extension: Some("jpg".into()),
                style: Style::LongForm
            }
        );
    }
}
