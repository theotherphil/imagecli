//! Basic helper functions for working with markdown.

use std::fmt::Write;

/// A link within a markdown document
pub fn markdown_internal_link(section: &str) -> String {
    format!("[{}](#{})", section, markdown_anchor_name(section))
}

/// Name of the anchor github generate for a markdown section.
fn markdown_anchor_name(section: &str) -> String {
    section.trim().replace(",", "").replace(" ", "-").to_lowercase()
}

/// Writes a table for inclusion in github-flavoured-markdown file.
pub fn markdown_table<'a, 'b, T: 'a, F, I>(
    header: &'b [&'static str],
    rows: I,
    render_row: F
) -> String
where
    I: Iterator<Item=&'a T>,
    F: Fn(&'a T) -> Vec<String>
{
    let mut table = String::new();
    writeln!(table, "{}", header.join("|")).unwrap();
    let divider: Vec<_> = std::iter::repeat("---").take(header.len()).collect();
    writeln!(table, "{}", divider.join("|")).unwrap();
    for row in rows {
        writeln!(table, "{}", render_row(row).join("|")).unwrap();
    }
    table
}

/// A markdown header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Header {
    pub level: usize,
    pub content: String,
}

pub fn find_headers(contents: &str) -> Vec<Header> {
    contents.split("\n").filter_map(header).collect()
}

pub fn header(line: &str) -> Option<Header> {
    let mut count = 0;
    for char in line.chars().skip_while(|c| c.is_whitespace()) {
        if char == '#' {
            count += 1;
            continue;
        }
        break;
    }
    if count == 0 {
        None
    } else {
        Some(Header {
            level: count,
            content: (&line[count..]).trim().into(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn test_markdown_table() {
        struct Row {
            l: &'static str,
            m: &'static str,
            r: &'static str,
        }
        let rows = vec![
            Row { l: "a00", m: "a01", r: "a02" },
            Row { l: "a10", m: "a11", r: "a12" },
        ];
        assert_eq!(
            markdown_table(
                &["First", "Second", "Third"],
                rows.iter(),
                |row: &Row| vec![row.l.into(), row.m.into(), row.r.into()]
            ),
            "First|Second|Third\n---|---|---\na00|a01|a02\na10|a11|a12\n"
        );
    }

    #[test]
    fn test_header() {
        assert_eq!(
            header("Foo"),
            None
        );
        assert_eq!(
            header("# Foo"),
            Some(Header { level: 1, content: "Foo".into() })
        );
        assert_eq!(
            header("## Foo"),
            Some(Header { level: 2, content: "Foo".into() })
        );
    }

    #[test]
    fn test_markdown_internal_link() {
        assert_eq!(
            markdown_internal_link("Some section, with a comma"),
            "[Some section, with a comma](#some-section-with-a-comma)"
        );
    }
}
