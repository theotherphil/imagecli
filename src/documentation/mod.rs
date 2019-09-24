//! Functions for generating documentation for this application.

pub mod markdown;
pub mod stack_diagram;
pub mod template;

use crate::error::{IoError, Result};
use crate::image_ops::{documentation, Alias, Documentation};
use markdown::{
    find_headers, header, markdown_anchor_name, markdown_internal_link, markdown_table, Header,
};
use snafu::ResultExt;
use std::fmt::Write;
use template::{parse_diagram, parse_example};

/// Name of directory to use as output for examples in this section.
fn dir_name(section: &str) -> String {
    section
        .trim()
        .replace(",", "")
        .replace(" ", "-")
        .to_lowercase()
}

/// Reads README_template.md and writes README.md.
pub fn generate_readme(verbose: bool) -> Result<()> {
    let read_path = "README_template.txt";
    let write_path = "README.md";

    let template = std::fs::read_to_string(read_path).context(IoError {
        context: format!("Unable to read file '{}", read_path),
    })?;

    let rendered = render_readme(&template, verbose)?;

    std::fs::write(write_path, rendered).context(IoError {
        context: format!("Unable to write to file '{}'", write_path),
    })?;

    Ok(())
}

fn render_readme(template: &str, verbose: bool) -> Result<String> {
    let mut result = String::new();
    let mut examples = Vec::new();
    let mut current_section = Header {
        level: 0,
        title: "root".into(),
    };
    let mut current_section_example_count = 0;

    let mut lines = template.lines();
    while let Some(line) = lines.next() {
        match line.trim() {
            "$EXAMPLE(" => {
                let example = parse_example(&mut lines);
                let instantiated = example.instantiate(
                    "images".into(),
                    format!("images/{}", dir_name(&current_section.title)),
                    format!("ex{}", current_section_example_count),
                );
                write!(result, "{}", instantiated.render_for_documentation())?;
                examples.push(instantiated);
                current_section_example_count += 1;
            }
            "$STACK_DIAGRAM(" => {
                let diagram = parse_diagram(&mut lines);
                result.push_str(&diagram.render_for_markdown());
            }
            _ => {
                if let Some(header) = header(line) {
                    current_section = header;
                    current_section_example_count = 0;
                }
                writeln!(result, "{}", line)?;
            }
        }
    }

    for example in &examples {
        if verbose {
            println!(
                "Running example\n\t'{}'",
                example.command_line_for_documentation()
            );
        }
        example.run(verbose)?;
    }

    let mut operations = String::new();

    // Summary table
    let mut summaries = Vec::new();
    for doc in documentation().iter() {
        summaries.push(summary_from_documentation(doc));
        for alias in &doc.aliases {
            summaries.push(summary_from_alias(alias, doc.operation));
        }
    }
    summaries.sort_by_key(|s| s.name);
    let operations_table = markdown_table(
        &["Operation", "Usage", "Description"],
        summaries.iter(),
        row_from_summary,
    );

    writeln!(operations, "{}", operations_table)?;

    // Detailed per-op documentation
    for docs in documentation().iter() {
        writeln!(operations, "\n### {}\n", docs.operation)?;
        writeln!(operations, "Usage: `{}`\n", docs.usage)?;
        writeln!(operations, "{}", docs.explanation)?;
        for alias in &docs.aliases {
            writeln!(operations, "\n##### Alias: {}", alias.name)?;
            writeln!(operations, "\nUsage: `{}`\n", alias.usage)?;
            writeln!(operations, "{}", alias.description)?;
        }
        if !docs.examples.is_empty() {
            writeln!(operations, "\n#### Examples\n")?;
            for (count, example) in docs.examples.iter().enumerate() {
                let instantiated = example.instantiate(
                    "images".into(),
                    "images/operations".into(),
                    format!("{}_{}", docs.operation, count),
                );
                write!(operations, "{}", instantiated.render_for_documentation())?;
                instantiated.run(verbose)?;
            }
        }
    }

    let result = result.replace("$OPERATIONS", &operations);

    let headers = find_headers(&result);
    let (min_level, max_level) = (2, 2);

    let mut toc = Vec::new();
    for header in headers
        .iter()
        .filter(|h| (h.level >= min_level) && (h.level <= max_level))
    {
        let prefix = " ".repeat(2 * (header.level - min_level));
        toc.push(format!(
            "{} - {}",
            prefix,
            markdown_internal_link(&header.title)
        ));
    }

    let toc = toc.join("\n");
    let result = result.replace("$TABLE_OF_CONTENTS", &toc);

    Ok(result)
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Summary {
    name: &'static str,
    usage: &'static str,
    explanation: String,
    alias_for: Option<&'static str>,
}

fn summary_from_documentation(doc: &Documentation) -> Summary {
    Summary {
        name: doc.operation,
        usage: doc.usage,
        explanation: doc.explanation.into(),
        alias_for: None,
    }
}

fn summary_from_alias(alias: &Alias, operation: &'static str) -> Summary {
    Summary {
        name: alias.name,
        usage: alias.usage,
        explanation: format!("See {}.", markdown_internal_link(operation)),
        alias_for: Some(operation),
    }
}

fn row_from_summary(summary: &Summary) -> Vec<String> {
    vec![
        format!(
            "[{}](#{})",
            summary.name,
            markdown_anchor_name(summary.alias_for.unwrap_or(&summary.name))
        ),
        format!("`{}`", summary.usage.replace("|", "\\|")),
        summary.explanation.split('\n').nth(0).unwrap().into(),
    ]
}
