//! Functions for generating documentation for this application.

pub mod markdown;
pub mod stack_diagram;
pub mod template;

use crate::error::Result;
use crate::image_ops::documentation;
use markdown::{find_headers, header, markdown_internal_link, markdown_table, Header};
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
pub fn generate_readme() -> Result<()> {
    let template = std::fs::read_to_string("README_template.txt")?;
    let rendered = render_readme(&template)?;
    std::fs::write("README.md", rendered)?;
    Ok(())
}

fn render_readme(template: &str) -> Result<String> {
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
        example.run();
    }

    let mut operations = String::new();

    // Summary table
    let operations_table = markdown_table(
        &["Operation", "Usage", "Description"],
        documentation().iter(),
        |doc| {
            vec![
                markdown_internal_link(&doc.operation),
                format!("`{}`", doc.usage.replace("|", "\\|")),
                doc.explanation.split("\n").nth(0).unwrap().into(),
            ]
        },
    );
    writeln!(operations, "{}", operations_table)?;

    // Detailed per-op documentation
    for docs in documentation().iter() {
        writeln!(operations, "\n### {}\n", docs.operation)?;
        writeln!(operations, "Usage: `{}`\n", docs.usage)?;
        writeln!(operations, "{}", docs.explanation)?;
        if docs.examples.len() > 0 {
            writeln!(operations, "\n#### Examples\n")?;
            for (count, example) in docs.examples.iter().enumerate() {
                let instantiated = example.instantiate(
                    "images".into(),
                    "images/operations".into(),
                    format!("{}_{}", docs.operation, count),
                );
                write!(operations, "{}", instantiated.render_for_documentation())?;
                instantiated.run();
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
